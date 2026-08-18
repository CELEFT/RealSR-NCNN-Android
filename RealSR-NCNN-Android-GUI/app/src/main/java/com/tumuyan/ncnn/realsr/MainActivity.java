package com.tumuyan.ncnn.realsr;

import static com.tumuyan.ncnn.realsr.UriUntils.getFileName;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.ClipData;
import android.content.Context;
import android.content.ComponentName;
import android.content.ServiceConnection;
import android.content.Intent;
import android.app.PendingIntent;
import android.os.IBinder;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.icu.text.SimpleDateFormat;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.SearchView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.davemorrissey.labs.subscaleview.ImageSource;
import com.davemorrissey.labs.subscaleview.SubsamplingScaleImageView;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class MainActivity extends AppCompatActivity {
    private static final int SELECT_IMAGE = 1, SELECT_MULTI_IMAGE = 2;
    private static final int MY_PERMISSIONS_REQUEST = 100;
    private static final String CMD_CP_LIB_OPENCL = " if [ -e /system/vendor/lib64/libOpenCL.so ]; then cp /system/vendor/lib64/libOpenCL.so ./; elif [ -e /system/lib64/libOpenCL.so ]; then cp /system/lib64/libOpenCL.so ./; elif  [ -e /system/vendor/lib/libOpenCL.so ]; then cp /system/vendor/lib/libOpenCL.so ./; elif [ -e /system/lib/libOpenCL.so ]; then cp /system/lib/libOpenCL.so ./; else echo \"[warning]libOpenCL.so not find\"; fi; if [ -e /system/vendor/lib/egl/libGLES_mali.so ]; then cp /system/vendor/lib/egl/libGLES_mali.so ./; elif [ -e /system/lib/egl/libGLES_mali.so ]; then cp /system/lib/egl/libGLES_mali.so ./; else echo \"[warning]libGLES_mali.so not find\"; fi";
    private static final String CMD_RESET_CACHE = CMD_CP_LIB_OPENCL
            + ";rm -f *.cache;rm -f */*.cache;chmod +x *; echo Cache has been reset.;ls";
    private int selectCommand = 0;
    private String threadCount = "";
    private SubsamplingScaleImageView imageView;
    private TextView logTextView;
    private boolean initProcess;
    private final String galleryPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)
            + File.separator + "RealSR";
    private File outputFile, outputGif, inputFile, titleFile;

    /**
     * String dir 是应用的工作目录，用于存放临时文件和执行命令。
     * 典型值： /data/data/com.tumuyan.ncnn.realsr/cache/realsr
     * 在 onCreate() 阶段赋值，后续无法修改: dir = cache_dir + "/realsr";
     * 其他应用无法访问或修改，使用此变量不存在注入风险。
     */
    private String dir, cache_dir;
    private String modelName = "SR";
    private SearchView searchView;
    private MenuItem menuProgress;
    private Spinner spinner;
    private boolean newTask;
    private int format, name, name2, notify, dirOutputFormat;
    private String BUSY, ERR, DONE;
    private String outputSavePath = "";
    private String inputFileName = "";

    private String[] formats;

    private String[] command = null;
    private String log = "";
    private CommandListManager commandListManager;
    private ProgressLogHelper progressLogHelper;

    private final String[] bench_mark_commands = new String[] {
            "./realsr-ncnn -c 46 -i img/PM5544.jpeg -o input.png  -m models-Real-ESRGAN",
            "./realsr-ncnn -c 46 -i input.png -o output.png  -m models-Real-ESRGANv3-anime -s 4"
    };
    private int tileSize;
    private boolean useCPU;
    private int mnnBackend;
    private boolean keepScreen;
    private boolean useMultFiles;
    private boolean prePng;
    private boolean preFrame;
    private boolean autoSave;
    private boolean showSearchView, showFinalCommand;
    private String savePath = galleryPath;

    private static final int NOTIFY_ID = 1;
    private static final String CHANNEL_ID_RESULT = "channel_result";

    private void sendNotification(Context mContext, String text, boolean force) {
        // New Logic: 0=Silent, 1=Result, 2=Detailed, 3=Detailed(AutoDismiss).
        // So if notify == 0 or 3, we don't show result notification.
        // But if force is true, we show it anyway (e.g. for error in AutoDismiss mode).
        if (!force && (notify == 0 || notify == 3))
            return;

        NotificationManager notificationManager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);

        if (text == null) {
            notificationManager.cancel(NOTIFY_ID);
            return;
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID_RESULT,
                    getString(R.string.notification_channel_result),
                    NotificationManager.IMPORTANCE_HIGH);
            channel.setDescription("Shows result of image processing tasks");
            notificationManager.createNotificationChannel(channel);
        }

        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT
                        | (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0));

        NotificationCompat.Builder mBuilder = new NotificationCompat.Builder(mContext, CHANNEL_ID_RESULT);
        mBuilder.setContentTitle(getString(R.string.app_name))
                .setContentText(text)
                .setWhen(System.currentTimeMillis())
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)
                .setDefaults(Notification.DEFAULT_SOUND | Notification.DEFAULT_VIBRATE);
        Notification notification = mBuilder.build();
        notificationManager.notify(NOTIFY_ID, notification); // Use same ID as Service to update/replace it
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        menuProgress = menu.findItem(R.id.progress);
        if (initProcess) {
            initProcess = false;
            menuProgress.setTitle("");
            Log.i("onCreateOptionsMenu", "onCreate() done");
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {

        final String q;
        String imageName = "/output.png";
        boolean bench_mark_mode = false;
        int v = item.getItemId();
        if (v == R.id.progress) {
            stopCommand();
            return false;
        } else if (v == R.id.menu_share) {
            if (inputIsGifAnimation)
                shareImage("output.gif");
            else
                shareImage("output.png");
            return false;
        } else if (v == R.id.menu_avir2) {
            q = "./resize-ncnn -i input.png -o output.png  -m avir -s 0.5";
        } else if (v == R.id.menu_nearest4) {
            q = "./resize-ncnn -i input.png -o output.png  -m nearest -s 4";
        } else if (v == R.id.menu_de_nearest) {
            q = "./resize-ncnn -i input.png -o output.png  -m de-nearest";
        } else if (v == R.id.menu_de_nearest2) {
            q = "./resize-ncnn -i input.png -o output.png  -m de-nearest2";
        } else if (v == R.id.menu_perfectpixel){
            q = "./resize-ncnn -i input.png -o output.png  -m perfectpixel -s 0";
        } else if (v == R.id.menu_perfectpixel1){
            q = "./resize-ncnn -i input.png -o output.png  -m perfectpixel -s 1";
        } else if (v == R.id.menu_perfectpixel2) {
            q = "./resize-ncnn -i input.png -o output.png  -m perfectpixel -s 5";
        } else if (v == R.id.menu_magick2) {
            q = "./magick input.png -resize 50% output.png";
        } else if (v == R.id.menu_magick3) {
            q = "./magick input.png -resize 33.33% output.png";
        } else if (v == R.id.menu_magick4) {
            q = "./magick input.png -resize 25% output.png";
        } else if (v == R.id.menu_out2in) {
            if (inputIsGifAnimation) {
                Toast.makeText(this, R.string.not_support_animation, Toast.LENGTH_SHORT).show();
                return false;
            } else {
                q = "cp output.png input.png";
                imageName = "/input.png";
            }
        } else if (v == R.id.menu_in) {
            q = "in";
        } else if (v == R.id.menu_out) {
            q = "out";
        } else if (v == R.id.menu_help) {
            q = "help";
        } else if (v == R.id.menu_reset_cache) {
            q = CMD_RESET_CACHE;
            imageName = "";
        } else if (v == R.id.menu_bench_mark) {
            String append_param = "";
            if (tileSize > 0)
                append_param = " -t " + tileSize;
            if (useCPU)
                append_param += (" -g -1");

            append_param += ";";
            q = "rm -rf *.png; ls *.png; " + bench_mark_commands[0] + append_param + bench_mark_commands[1]
                    + append_param;

            imageName = "/img/realsr.png";
            bench_mark_mode = true;
            imageView.setVisibility(View.GONE);
            if (keepScreen) {
                logTextView.setKeepScreenOn(true);
            }
        } else if (v == R.id.menu_dir_batch) {
            Intent intent = new Intent(this, DirectoryProcessActivity.class);
            startActivity(intent);
            return true;
        } else
            q = "";

        if (!run_fake_command(q)) {
            stopCommand();
            String finalImageName = imageName;
            boolean final_bench_mark_mode = bench_mark_mode;
            new Thread(() -> {
                if (q.equals(CMD_RESET_CACHE)) {
                    AssetsCopyer.releaseAssets(this, "realsr", cache_dir, false);
                }

                run20(q, final_bench_mark_mode, false);
                final File finalfile = new File(dir + finalImageName);
                if (finalfile.exists() && (!finalfile.isDirectory())) {
                    runOnUiThread(() -> {
                        imageView.setVisibility(View.VISIBLE);
                        imageView.setImage(ImageSource.uri(finalfile.getAbsolutePath()));
                        logTextView.setKeepScreenOn(false);
                    });
                } else {
                    runOnUiThread(() -> imageView.setVisibility(View.GONE));
                }
            }).start();
        }

        return super.onOptionsItemSelected(item);
    }

    // 删除文件或者目录
    public static void deleteFile(File f) {
        if (f.isDirectory()) {
            // 获取目录下所有文件和目录
            File[] files = f.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        deleteFile(file);
                    } else {
                        file.delete();
                    }
                }
            }
        }
        f.delete();
    }

    public void shareImage(String path) {
        Intent share_intent = new Intent();

        Uri contentUri = null;
        File file = null;
        if (!outputSavePath.isEmpty()) {
            file = new File(outputSavePath);
            if (file.exists()) {
                contentUri = FileProvider.getUriForFile(this,
                        BuildConfig.APPLICATION_ID + ".fileprovider",
                        file);
            }
        }

        if (contentUri == null) {
            file = new File(dir, path);
            if (file.exists()) {
                contentUri = FileProvider.getUriForFile(this,
                        BuildConfig.APPLICATION_ID + ".fileprovider",
                        file);
            }
        }

        if (contentUri != null) {
            String suffix = file.getName().replaceFirst(".+\\.([^.]+)$", "$1").toLowerCase(Locale.ROOT);
            switch (suffix) {
                case "png":
                    share_intent.setType("image/png");
                    break;
                case "jpg":
                    share_intent.setType("image/jpg");
                    break;
                case "webp":
                    share_intent.setType("image/webp");
                    break;
                case "heif":
                    share_intent.setType("image/heif");
                    break;
                case "gif":
                    share_intent.setType("image/gif");
                    break;
                default:
                    share_intent.setType("image/*");
                    break;
            }

            share_intent.setAction(Intent.ACTION_SEND);// 设置分享行为
            share_intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            share_intent.putExtra(Intent.EXTRA_STREAM, contentUri);
            Log.i("shareImage()", "uri = " + contentUri);
            startActivity(Intent.createChooser(share_intent, "Share"));

        } else {
            Toast.makeText(getApplicationContext(), R.string.output_not_exits, Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    public void onResume() {
        super.onResume();

        formats = getResources().getStringArray(R.array.format);
        BUSY = getResources().getString(R.string.busy);
        ERR = getString(R.string.notification_fail);
        DONE = getString(R.string.done);

        SharedPreferences mySharePerferences = getSharedPreferences("config", Activity.MODE_PRIVATE);
        tileSize = mySharePerferences.getInt("tileSize", 0);
        threadCount = mySharePerferences.getString("threadCount", "");
        keepScreen = mySharePerferences.getBoolean("keepScreen", false);

        useMultFiles = mySharePerferences.getBoolean("useMultFiles", false);
        prePng = mySharePerferences.getBoolean("PrePng", true);
        preFrame = mySharePerferences.getBoolean("PreFrame", true);
        useCPU = mySharePerferences.getBoolean("useCPU", false);
        mnnBackend = mySharePerferences.getInt("mnnBackend", 3);
        autoSave = mySharePerferences.getBoolean("autoSave", false);
        showSearchView = mySharePerferences.getBoolean("showSearchView", false);
        if (showSearchView)
            searchView.setVisibility(View.VISIBLE);
        else
            searchView.setVisibility(View.GONE);

        showFinalCommand = mySharePerferences.getBoolean("showFinalCommand", false) && showSearchView;

        notify = mySharePerferences.getInt("notify", 0);

        format = mySharePerferences.getInt("format", 0);
        dirOutputFormat = mySharePerferences.getInt("dirOutputFormat", 0);
        name = mySharePerferences.getInt("name", 0);
        name2 = mySharePerferences.getInt("name2", 0);

        // 构建命令列表（使用 CommandListManager）
        String[] presetLabels = getResources().getStringArray(R.array.style_array);
        boolean useCustomLabel = mySharePerferences.getBoolean("useCustomLabel", false);
        commandListManager = new CommandListManager(presetLabels,
                mySharePerferences.getString("extraPath", "").trim(),
                mySharePerferences.getString("extraCommand", "").trim(),
                mySharePerferences.getString("classicalFilters", getString(R.string.default_classical_filters))
                        .split("\\s+"),
                mySharePerferences.getString("magickFilters", getString(R.string.default_magick_filters))
                        .split("\\s+"));
        commandListManager.loadCustomLabels(mySharePerferences.getString("customLabels", ""));

        Set<String> hiddenPrograms = mySharePerferences.getStringSet("hiddenPrograms", new HashSet<String>());
        command = commandListManager.getFilteredCommands(hiddenPrograms);
        String[] displayLabels = commandListManager.getFilteredLabels(hiddenPrograms, useCustomLabel);

        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, displayLabels);
        spinner.setAdapter(adapter);

        if (selectCommand >= command.length)
            selectCommand = Math.max(0, command.length - 1);
        spinner.setSelection(selectCommand);

        savePath = mySharePerferences.getString("savePath", "");
        if (savePath.isEmpty())
            savePath = galleryPath;
        try {
            File file = new File(savePath);
            if (file.isFile())
                file.delete();
            if (!file.exists())
                file.mkdirs();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void readFileFromShare() {
        Intent intent = getIntent();
        String action = intent.getAction();

        if (Intent.ACTION_SEND.equals(action)) {
            deleteFile(inputFile);
            Uri uri = intent.getParcelableExtra(Intent.EXTRA_STREAM);
            inputFileName = getFileName(uri, this);
            assert inputFileName != null;
            inputFileName = inputFileName.replaceFirst("\\.[^.]+$", "");
            Log.i("input file name", inputFileName);
            whiteFileFromUri(uri, "");

        } else if (Intent.ACTION_SEND_MULTIPLE.equals(action)) {
            ArrayList<Uri> imageUris = intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM);
            handleSelectedImages(imageUris);

        }
    }

    private boolean whiteFileFromUri(Uri uri, String path) {
        if (uri != null) {
            try {
                InputStream in = getContentResolver().openInputStream(uri);
                if (null != in)
                    saveInputImage(in, path);
                else
                    Toast.makeText(this, R.string.share_is_null, Toast.LENGTH_SHORT).show();
                return true;
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return false;
    }

    private ProcessingService processingService;
    private boolean isBound = false;

    private ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className, IBinder service) {
            ProcessingService.LocalBinder binder = (ProcessingService.LocalBinder) service;
            processingService = binder.getService();
            isBound = true;
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            isBound = false;
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (BuildConfig.DEBUG) {
            setTitle(getString(R.string.app_name) + " (Debug)");
        }

        Intent serviceIntent = new Intent(this, ProcessingService.class);
        bindService(serviceIntent, connection, Context.BIND_AUTO_CREATE);

        imageView = findViewById(R.id.photo_view);
        logTextView = findViewById(R.id.tv_log);
        searchView = findViewById(R.id.serarch_view);

        SharedPreferences mySharePerferences = getSharedPreferences("config", Activity.MODE_PRIVATE);
        prePng = mySharePerferences.getBoolean("PrePng", true);
        preFrame = mySharePerferences.getBoolean("PreFrame", true);

        int version = mySharePerferences.getInt("version", 0);
        String defaultCommand = mySharePerferences.getString("defaultCommand", "");
        searchView.setQuery(defaultCommand, false);

        cache_dir = this.getCacheDir().getAbsolutePath();
        AssetsCopyer.releaseAssets(this, "realsr", cache_dir, version == BuildConfig.VERSION_CODE);

        SharedPreferences.Editor editor = mySharePerferences.edit();
        editor.putInt("version", BuildConfig.VERSION_CODE);
        editor.apply();

        int orientation = mySharePerferences.getInt("ORIENTATION", 0);
        if (orientation == 1) {
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_SENSOR);
        } else if (orientation == 2)
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        else if (orientation == 3) {
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        }

        dir = cache_dir + "/realsr";

        outputFile = new File(dir, "output.png");
        outputGif = new File(dir, "output.gif");
        inputFile = new File(dir, "input.png");
        titleFile = new File(dir, "img/realsr.png");
        showImage(titleFile, getString(R.string.default_log));

        if (version != BuildConfig.VERSION_CODE) {
            // dir 来自应用缓存目录，参数来源可信
            run_command("cd " + dir + ";" + CMD_CP_LIB_OPENCL + " chmod +x *");
        } else {
            run_command("chmod +x " + dir + " -R");
        }

        spinner = findViewById(R.id.spinner);
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                selectCommand = pos;
                Log.i("setOnItemSelectedListener", "select " + pos);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        selectCommand = mySharePerferences.getInt("selectCommand", 2);

        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {

                String q = searchView.getQuery().toString().trim();

                if (!run_fake_command(q)) {
                    stopCommand();
                    // 使用 ImageProcessor 执行
                    run20(query, false, true);
                }
                return false;
            }

            // 用户输入字符时激发该方法
            @Override
            public boolean onQueryTextChange(String newText) {
                if (newText.trim().length() < 2) {
                    if (menuProgress != null)
                        menuProgress.setTitle("");
                    return true;
                }
                if (imageView.getVisibility() == View.VISIBLE)
                    imageView.setVisibility(View.GONE);
                return true;
            }
        });
        findViewById(R.id.btn_open).setOnClickListener(view -> {
            if (useMultFiles) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                startActivityForResult(intent, SELECT_MULTI_IMAGE);

            } else {

                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });

        findViewById(R.id.btn_save).setOnClickListener(view -> {
            File f = inputIsGifAnimation ? outputGif : outputFile;

            if (!f.exists()) {
                Toast.makeText(this, R.string.output_not_exits, Toast.LENGTH_SHORT).show();
                return;
            } else if (f.isDirectory()) {
                File[] files = f.listFiles();
                if (files == null || files.length == 0) {
                    Toast.makeText(this, R.string.output_not_exits, Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(this, R.string.output_is_dir, Toast.LENGTH_SHORT).show();
                }
                return;
            }
            run_command(saveOutputCmd());
            checkSaveOutput();
        });

        findViewById(R.id.btn_run).setOnClickListener(view -> {
            menuProgress.setTitle("");
            {
                stopCommand();
                log = "";
                StringBuffer cmd;

                if (selectCommand >= command.length) {

                    cmd = new StringBuffer(spinner.getSelectedItem().toString());
                    Log.w("btn_run.onClick", "select=" + selectCommand + ", length=" + command.length + " text=" + cmd);

                    if (run_fake_command(cmd.toString()))
                        return;
                } else {
                    final String cmd_head = command[selectCommand];
                    cmd = new StringBuffer(cmd_head);
                    if (cmd_head.matches("./(realsr|srmd|waifu2x|realcugan|mnnsr)-ncnn.+")) {
                        if (tileSize > 0 && !cmd_head.contains(" -t "))
                            cmd.append(" -t ").append(tileSize);
                        if (!threadCount.isEmpty() && !cmd_head.contains(" -j "))
                            cmd.append(" -j ").append(threadCount);
                        if (useCPU && !cmd_head.startsWith("./srmd") && !cmd_head.startsWith("./mnnsr")
                                && !cmd_head.contains(" -g "))
                            cmd.append(" -g -1");
                        if (cmd_head.startsWith("./mnnsr") && !cmd_head.contains(" -b ")) {
                            cmd.append(" -b ").append(mnnBackend);
                        }
                    }
                }

                deleteFile(outputFile);
                if (inputIsGifAnimation) {
                    outputGif.delete();
                    outputFile.mkdir();
                }
                if (keepScreen) {
                    logTextView.setKeepScreenOn(true);
                }

                if (showFinalCommand) {
                    searchView.setQuery(cmd.toString(), false);
                    Toast.makeText(this, cmd.toString(), Toast.LENGTH_SHORT).show();
                }

                run20(cmd.toString(), false, true);
            }
        });

        findViewById(R.id.btn_setting).setOnClickListener(view -> {
            Intent intent = new Intent(this, SettingActivity.class);
            this.startActivity(intent);
            overridePendingTransition(0, android.R.anim.slide_out_right);
        });

        requirePremision();

        if (menuProgress != null)
            menuProgress.setTitle("");
        else
            initProcess = true;

        readFileFromShare();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (isBound) {
            unbindService(connection);
            isBound = false;
        }
    }

    private void requirePremision() {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[] { Manifest.permission.WRITE_EXTERNAL_STORAGE },
                    MY_PERMISSIONS_REQUEST);

        } else {
            // 权限已经被授予，在这里直接写要执行的相应方法即可
            File file = new File(savePath);
            if (file.isFile())
                file.delete();
            if (!file.exists())
                file.mkdirs();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
            @NonNull int[] grantResults) {
        if (requestCode == MY_PERMISSIONS_REQUEST) {
            if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(MainActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    // 处理选中的多个文件
    private void handleSelectedImages(List<Uri> uris) {
        if (uris == null || uris.isEmpty())
            return;
        deleteFile(inputFile);
        if (uris.size() == 1) {
            Uri url = uris.get(0);
            {

                inputFileName = getFileName(url, this).replaceFirst("\\.[^.]+$", "");
                Log.i("input file name", inputFileName);
                InputStream in;

                try {
                    in = getContentResolver().openInputStream(url);
                    if (null != in)
                        saveInputImage(in, "");
                    else
                        Toast.makeText(this, "input == null", Toast.LENGTH_SHORT).show();
                } catch (Exception e) {
                    e.printStackTrace();
                    return;
                }
            }
            return;
        }

        inputFile.mkdirs();
        outputFile.delete();

        SimpleDateFormat f = new SimpleDateFormat("MMdd_HHmmss");
        String time = f.format(new Date());
        for (int i = 0; i < uris.size(); i++) {
            Uri uri = uris.get(i);
            inputFileName = getFileName(uri, this).replaceFirst("\\.[^.]+$", "");
            switch (name2) {
                case 0:
                    inputFileName = String.format("%s_%s", inputFileName, time);
                    break;
                case 1:
                    inputFileName = String.format("%s_%d", inputFileName, i);
                    break;
                case 2:
                    inputFileName = String.format("%s_%d", time, i);
                    break;
                case 3:
                    inputFileName = time + "_" + inputFileName;
                    break;
            }
            String inputFilePath = String.format("%s/input.png/%s.png", dir, inputFileName);
            int j = 0;
            while (new File(inputFilePath).exists()) {
                j++;
                inputFilePath = dir + "/input.png/" + inputFileName + "_" + j + ".png";
            }
            whiteFileFromUri(uri, inputFilePath);
        }
        int inputFileSize = inputFile.listFiles().length;
        logTextView.setText(String.format(getString(R.string.input_file_size), inputFileSize));
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        if (resultCode == RESULT_OK && null != data) {
            Uri url = data.getData();

            if (requestCode == SELECT_IMAGE && null != url) {
                deleteFile(inputFile);
                inputFileName = getFileName(url, this).replaceFirst("\\.[^.]+$", "");
                Log.i("input file name", inputFileName);
                InputStream in;

                try {
                    in = getContentResolver().openInputStream(url);
                    if (null != in)
                        saveInputImage(in, "");
                    else
                        Toast.makeText(this, "input == null", Toast.LENGTH_SHORT).show();
                } catch (Exception e) {
                    e.printStackTrace();
                    return;
                }
            } else if (requestCode == SELECT_MULTI_IMAGE) {
                List<Uri> imageUris = new ArrayList<>();
                ClipData clipData = data.getClipData();
                for (int i = 0; i < clipData.getItemCount(); i++) {
                    imageUris.add(clipData.getItemAt(i).getUri());
                }
                handleSelectedImages(imageUris);
            }

        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    // 在主进程执行命令但是不刷新UI，也不被打断
    public int get_gif_frame_delay(@NonNull String path) {

        StringBuilder con = new StringBuilder();
        String result;

        try {
            ProcessBuilder processBuilder = new ProcessBuilder("sh");
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();

            OutputStream os = process.getOutputStream();
            // dir 来自应用缓存目录，参数来源可信；path 为用户文件需转义
            String cmd = "cd " + dir + "; export LD_LIBRARY_PATH=" + dir
                    + "; ./magick identify -format \"%T \" " + ShellUtils.escapeShellArgument(path) + " ";
            os.write((cmd + "\n").getBytes());
            os.write("exit\n".getBytes());
            os.flush();
            os.close();

            BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));
            while ((result = br.readLine()) != null) {
                con.append(result);
                con.append('\n');
            }
            process.waitFor();

        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();

            Log.d("get_gif_frame_delay()", "crash; result=" + con);
            return -1;
        }

        String[] data = con.toString().strip().split("\\s+");
        if (data.length < 2)
            return 0;

        int avg = Integer.parseInt(data[1]);
        int dif = 0;
        for (String s : data) {
            dif += (Integer.parseInt(s) - avg);
        }
        avg = avg + dif / data.length;

        Log.d("get_gif_frame_delay()", "finish; result=" + con);
        return avg;
    }

    // 在主进程执行命令但是不刷新UI，也不被打断
    public boolean run_command(@NonNull String command) {

        if (command.trim().length() < 1) {
            Log.d("run_command", "command=" + command + "; break");
            return false;
        }

        StringBuilder con = new StringBuilder();
        String result;

        try {
            ProcessBuilder processBuilder = new ProcessBuilder("sh");
            processBuilder.redirectErrorStream(true);

            Process process = processBuilder.start();

            OutputStream os = process.getOutputStream();
            if (command.startsWith("./magick")) {
                // dir 来自应用缓存目录，参数来源可信
                String magickCmd = "cd " + dir + "; export LD_LIBRARY_PATH=" + dir + "; " + command;
                os.write((magickCmd + "\n").getBytes());
            } else {
                os.write((command + "\n").getBytes());
            }
            os.write("exit\n".getBytes());
            os.flush();
            os.close();

            BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));
            while ((result = br.readLine()) != null) {
                con.append(result);
                con.append('\n');
            }

            process.waitFor();

        } catch (Exception e) {
            e.printStackTrace();
            Log.d("run_command", "command=" + command + "; crash; result=" + con);
            return false;
        }

        Log.d("run_command", "command=" + command + "; finish; result=" + con);
        return true;
    }

    private String progressText = "";

// 在 MainActivity 类中，替换原有的 run20 方法
public synchronized boolean run20(@NonNull String cmd, boolean bench_mark_mode, boolean sr) {
    newTask = false;
    Log.i("run20", "cmd = " + cmd);
    final long timeStart = System.currentTimeMillis();
    boolean export_dir = false;

    String finalCmd = cmd;

    // ---- 判断是否为支持目录模式的命令 ----
    if (cmd.startsWith("./realsr-ncnn") || cmd.startsWith("./mnnsr-ncnn")
            || cmd.startsWith("./srmd-ncnn") || cmd.startsWith("./realcugan-ncnn")
            || cmd.startsWith("./resize-ncnn") || cmd.startsWith("./waifu2x-ncnn")
            || cmd.startsWith("./magick input") || cmd.startsWith("./Anime4k")) {

        if (cmd.contains(" input.png ") && cmd.contains(" output.png")) {
            // ★★★ 关键：检测输入是否为目录（来自 GIF 拆帧或用户多选） ★★★
            if (inputFile.isDirectory()) {
                export_dir = true;
                String inputDirPath = inputFile.getAbsolutePath() + "/";
                String safeInputDir = ShellUtils.escapeShellArgument(inputDirPath);
                String safeOutputDir = ShellUtils.escapeShellArgument(savePath);

                // 替换占位符
                finalCmd = cmd.replace(" input.png ", " " + safeInputDir + " ")
                              .replace(" output.png ", " " + safeOutputDir + " ");

                // 针对不同后端添加输出格式指定参数（确保输出为 PNG）
                if ((cmd.startsWith("./realsr-ncnn") || cmd.startsWith("./mnnsr-ncnn")
                        || cmd.startsWith("./srmd-ncnn") || cmd.startsWith("./realcugan-ncnn")
                        || cmd.startsWith("./waifu2x-ncnn")) && !finalCmd.contains(" -f ")) {
                    finalCmd += " -f png";
                }
                if (cmd.startsWith("./Anime4k") && !finalCmd.contains(" -E ")) {
                    finalCmd += " -E .png";
                }
                // resize-ncnn 和 magick 通常不需要 -f，此处不处理
                Log.i("run20", "Directory mode enabled. finalCmd = " + finalCmd);
            } else {
                // ---- 单文件模式（保留原有逻辑，例如替换 output.png 到 savePath） ----
                // 如果你之前有单文件的 savePath 替换逻辑，可保留，这里不覆盖
                // 但为了统一，我们也可以让单文件也使用 savePath 作为输出路径
                // 但为了避免破坏已有功能，暂时保留原样，只改目录分支
                // 你可根据需要自行扩展
            }
        }
    }

    // 如果是基准测试模式，关闭自动保存
    final boolean run_ncnn = bench_mark_mode || !modelName.equals("SR");
    boolean export_one_file = run_ncnn && (autoSave || (inputFile.isDirectory() && inputIsGifAnimation))
            && cmd.contains("output.png");
    if (bench_mark_mode) {
        export_one_file = false;
        runOnUiThread(() -> {
            menuProgress.setTitle(BUSY);
            sendNotification(MainActivity.this, BUSY, false);
        });
    }
    final boolean save = export_one_file;

    // ---- 构造最终执行命令（添加保存操作） ----
    CommandBuilder builder = new CommandBuilder();
    builder.append(finalCmd);

    if (save) {
        String export_cmd = saveOutputCmd();
        if (inputIsGifAnimation)
            builder.append(";./magick -delay " + inputGifDelay + " output.png/* -loop 0 " + ShellUtils.escapeShellArgument(outputSavePath));
        else
            builder.append(";" + export_cmd);
    } else {
        outputSavePath = "";
    }

    final String executionCmd = builder.build();
    final String effectivelyFinalCmd = finalCmd;
    final boolean final_export_dir = export_dir;

    progressLogHelper = new ProgressLogHelper();

    // ---- 通过 Service 执行命令 ----
    if (isBound && processingService != null) {
        progressLogHelper.reset();
        processingService.startTask(executionCmd, dir, notify, new ImageProcessor.ProcessCallback() {
            @Override
            public void onProgress(String line) {
                progressLogHelper.appendLine(line);
                runOnUiThread(() -> {
                    logTextView.setText(progressLogHelper.getDisplayText());
                    if (progressLogHelper.hasProgress()) {
                        menuProgress.setTitle(progressLogHelper.getProgressText());
                    }
                });
            }

            @Override
            public void onCompleted(String result, boolean success) {
                String logResult = progressLogHelper.getCompletionSummary(success, modelName, run_ncnn);

                if (bench_mark_mode) {
                    logResult = logResult.replace("\n", String.format(", Benchmark run on %s\n%s",
                            DeviceInfo.getConfigStr(useCPU, tileSize), DeviceInfo.getInfo(MainActivity.this)));
                }

                progressLogHelper.appendLine(logResult);
                String finalLog = progressLogHelper.getFullLog();
                log = finalLog;

                runOnUiThread(() -> {
                    logTextView.setText(finalLog);
                    menuProgress.setTitle(success ? DONE : ERR);
                    boolean forceShow = !success && notify == 3;
                    sendNotification(MainActivity.this, success ? DONE : ERR, forceShow);

                    if (keepScreen) {
                        logTextView.setKeepScreenOn(false);
                    }

                    if (success) {
                        // ---- 处理 GIF 合成（若拆帧处理，需合成回 GIF） ----
                        if (inputIsGifAnimation && inputFile.isDirectory()) {
                            // 使用 ImageMagick 合成 GIF（后续可迁移至 FFmpeg）
                            String gifOutputPath = savePath + File.separator + modelName + ".gif";
                            String cmdGif = "./magick -delay " + inputGifDelay + " " 
                                    + ShellUtils.escapeShellArgument(savePath + "/*.png") 
                                    + " -loop 0 " + ShellUtils.escapeShellArgument(gifOutputPath);
                            new Thread(() -> {
                                run_command(cmdGif);
                                runOnUiThread(() -> {
                                    Toast.makeText(MainActivity.this, "GIF saved: " + gifOutputPath, Toast.LENGTH_LONG).show();
                                    // 可选：更新图片预览
                                    updateImage(gifOutputPath, getString(R.string.hr), false);
                                });
                            }).start();
                            // 注意：这里不阻塞主线程，因此后续 UI 更新可能与合成同时进行
                        }

                        // ---- 常规保存处理 ----
                        if (save) {
                            if (!outputFile.exists()) {
                                Toast.makeText(getApplicationContext(), R.string.output_not_exits, Toast.LENGTH_SHORT)
                                        .show();
                            } else {
                                checkSaveOutput();
                            }
                        } else if (final_export_dir) {
                            Toast.makeText(getApplicationContext(), R.string.save_succeed, Toast.LENGTH_SHORT).show();
                        }

                        if (!save && inputFile.isDirectory()) {
                            if (inputIsGifAnimation)
                                scanFiles(new String[]{outputSavePath});
                            else {
                                File[] files = inputFile.listFiles();
                                if (files != null) {
                                    List<String> outputPaths = new ArrayList<>();
                                    for (File file : files) {
                                        outputPaths.add(savePath + File.separator + file.getName());
                                    }
                                    scanFiles(outputPaths.toArray(new String[0]));
                                }
                            }
                        }

                        // ---- 显示结果图片 ----
                        boolean showImgView = (effectivelyFinalCmd.contains("output.png"));
                        if (showImgView) {
                            if (outputFile.exists() && outputFile.isFile()) {
                                updateImage(dir + "/output.png", String.format("%s\n%s", getString(R.string.hr), log),
                                        false);
                            } else if (inputIsGifAnimation && outputFile.exists() && outputFile.isDirectory()
                                    && outputFile.listFiles().length > 1) {
                                updateImage(outputFile.listFiles()[0].getPath(),
                                        String.format("%s\n%s", getString(R.string.hr), log), false);
                            } else {
                                updateImage(dir + "/input.png", String.format("%s\n%s", getString(R.string.lr), log),
                                        false);
                            }
                        }
                        if (!effectivelyFinalCmd.contains("output.png"))
                            imageView.setVisibility(View.GONE);
                    } else {
                        if (!effectivelyFinalCmd.contains("output.png"))
                            imageView.setVisibility(View.GONE);
                    }
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    logTextView.append("\nError: " + error);
                    sendNotification(MainActivity.this, ERR, true);
                    if (keepScreen) {
                        logTextView.setKeepScreenOn(false);
                    }
                });
            }
        });
    } else {
        Toast.makeText(this, "Service not bound", Toast.LENGTH_SHORT).show();
        return false;
    }

    return true;
}

    private void stopCommand() {
        if (isBound && processingService != null) {
            processingService.cancelTask();
            if (menuProgress != null)
                menuProgress.setTitle("");
        }
        newTask = true;
    }

    private boolean inputIsGifAnimation;
    private int inputGifDelay;

    /**
     * 保存文件
     *
     * @param in   输出的文件流
     * @param path 输出的文件路径，路径为空时保存为input.png
     * @return 是否保存成功
     */
private boolean saveInputImage(@NonNull InputStream in, String path) {
    Log.i("saveInputImage", "start");
    inputIsGifAnimation = false;
    boolean inputOneImage = false;

    // 单文件模式处理
    if (path == null || path.isEmpty()) {
        inputOneImage = true;
        path = dir + "/input.png";
    }
    File targetFile = new File(path);
    if (targetFile.exists()) {
        targetFile.delete();
    }

    // 临时文件
    File tempFile = new File(dir, "tmp");
    if (tempFile.exists()) tempFile.delete();

    try {
        tempFile.createNewFile();
        OutputStream outStream = new FileOutputStream(tempFile);
        byte[] buffer = new byte[4112];
        int read;

        // 读取首块数据（用于格式检测）
        if ((read = in.read(buffer)) != -1) {
            outStream.write(buffer, 0, read);
        } else {
            in.close();
            outStream.close();
            return false;
        }

        // 继续读取剩余数据
        while ((read = in.read(buffer)) != -1) {
            outStream.write(buffer, 0, read);
        }
        outStream.flush();
        outStream.close();
        in.close();

        // 检测文件头
        byte[] header = new byte[12];
        try (FileInputStream fis = new FileInputStream(tempFile)) {
            fis.read(header);
        }
        int match = PreprocessToPng.match(header);

        // ------- 核心处理逻辑 -------
        if (match == PreprocessToPng.TYPE_PNG) {
            // 情况1：PNG 直接复制（跳过转换）
            run_command("cp " + ShellUtils.escapeShellArgument(tempFile.getAbsolutePath()) + " " + ShellUtils.escapeShellArgument(path));
            Log.i("saveInputImage", "PNG file copied directly.");
        } else if (match == PreprocessToPng.TYPE_GIF && preFrame && inputOneImage) {
            // 情况2：GIF 动图（拆帧或转单帧，后续可迁移至 FFmpeg）
            inputGifDelay = get_gif_frame_delay(tempFile.getAbsolutePath());
            inputIsGifAnimation = inputGifDelay > 0;
            Log.i("inputGifDelay", "delay=" + inputGifDelay + ", isAnim=" + inputIsGifAnimation);

            if (inputIsGifAnimation) {
                // 多帧动画：拆帧到 input.png/ 目录
                deleteFile(inputFile);
                inputFile.mkdirs();
                run_command("./magick tmp -coalesce -delay 0 input.png/%04d.png");
                tempFile.delete();
                updateImage(inputFile.getAbsolutePath(), getString(R.string.lr), false);
                return true;
            } else {
                // 单帧 GIF：转 PNG
                run_command("./magick tmp " + ShellUtils.escapeShellArgument(path));
            }
        } else {
            // 情况3：其他所有静态图片（JPG, BMP, WEBP, HEIF, AVIF 等）统一转 PNG
            run_command("./magick tmp " + ShellUtils.escapeShellArgument(path));
            Log.i("saveInputImage", "Converted to PNG by ImageMagick.");
        }

        // 清理临时文件
        tempFile.delete();

    } catch (IOException e) {
        e.printStackTrace();
        return false;
    }

    updateImage(path, getString(R.string.lr), false);
    return true;
}
    private void updateImage(final String path, String text, boolean keepScreen) {
        Log.i("saveInputImage", "runOnUiThread");
        File file = new File(path);
        runOnUiThread(() -> {
            if (file.exists()) {
                if (file.isDirectory()) {
                    if (file.listFiles().length > 0) {
                        imageView.setVisibility(View.VISIBLE);
                        imageView.setImage(ImageSource.uri(file.listFiles()[0].getPath()));
                        Log.i("saveInputImage", "finish, directory");
                    } else {
                        imageView.setVisibility(View.GONE);
                        Log.i("saveInputImage", "finish, empty directory");
                    }
                    logTextView.setText(text);
                } else {
                    imageView.setVisibility(View.VISIBLE);
                    imageView.setImage(ImageSource.uri(path));
                    logTextView.setText(getImageResolation(file, text));
                    Log.i("saveInputImage", "finish, file");
                }

            } else
                Log.i("saveInputImage", "skip");
            if (keepScreen) {
                logTextView.setKeepScreenOn(false);
            }

        });
    }

    private boolean run_fake_command(String q) {
        if (q == null)
            return true;
        if (q.isEmpty())
            return true;
        if (q.equals("help")) {
            showImage(titleFile, getString(R.string.default_log));
        } else if (q.equals("in")) {
            showImage(inputFile, getString(R.string.lr));
        } else if (q.equals("out")) {
            showImage(outputFile, getString(R.string.hr));
        } else if (q.startsWith("show ")) {
            String path = q.replaceFirst("\\s*show\\s+(\\S+)\\s*", "$1");
            File file = new File(path);
            if (!file.exists()) {
                path = dir + "/" + path;
                file = new File(path);
            }
            showImage(file, getString(R.string.show) + path);

        } else if (q.equals("none")) {
            showImage(null, getString(R.string.menu_reset_cache));
        } else if (q.equals(CMD_RESET_CACHE)) {
            showImage(null, getString(R.string.menu_reset_cache) + "...");
            return false;
        } else
            return false;
        return true;
    }

    private void showImage(File file, String info) {
        if (file == null) {
            imageView.setVisibility(View.GONE);
            logTextView.setText(info);
        } else if (file.exists() && (!file.isDirectory())) {
            imageView.setVisibility(View.VISIBLE);
            imageView.setImage(ImageSource.uri(file.getAbsolutePath()));
            logTextView.setText(getImageResolation(file, info));
        } else if (file.isDirectory()) {
            imageView.setVisibility(View.GONE);
            File[] files = file.listFiles();
            if (files.length < 1) {
                logTextView.setText(getString(R.string.image_not_exists));
            } else
                logTextView.setText(getString(R.string.image_is_directory));
        } else {
            imageView.setVisibility(View.GONE);
            logTextView.setText(getString(R.string.image_not_exists));
        }
    }

    private static String getImageResolation(File file, String info) {
        if (info.trim().contains("\n"))
            return info;

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true; // 仅解码边界信息
        BitmapFactory.decodeFile(file.getAbsolutePath(), options);
        int width = options.outWidth;
        int height = options.outHeight;
        if (width > 0 && height > 0)
            return info + " " + width + "x" + height;
        return info;
    }

    /**
     * 通知android媒体库更新文件
     */
    private void scanFiles(String[] filePath) {
        Log.i("scanFiles()", "length=" + filePath.length);
        try {
            MediaScannerConnection.scanFile(getApplicationContext(), filePath, null,
                    (path, uri) -> {
                        Log.i("TAG", "Scanned " + path + ":");
                        Log.i("TAG", "-> uri=" + uri);
                    });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void checkSaveOutput() {
        File file = new File(outputSavePath);
        if (file.exists()) {
            Intent intent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
            Uri uri = Uri.fromFile(file);
            intent.setData(uri);
            sendBroadcast(intent);
            Toast.makeText(getApplicationContext(), R.string.save_succeed, Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(getApplicationContext(), R.string.save_fail, Toast.LENGTH_SHORT).show();
        }
    }

    // 生成输出图片的保存命令。采用"延迟转义"策略：在字符串构建阶段不进行转义，仅在最终返回时统一转义。
    private String saveOutputCmd() {

        SimpleDateFormat f = new SimpleDateFormat("MMdd_HHmmss");
        outputSavePath = savePath + File.separator;
        switch (name) {
            case 0:
                outputSavePath += modelName + "_" + f.format(new Date());
                break;
            case 1:
                outputSavePath += inputFileName + "_" + modelName + "_" + f.format(new Date());
                break;
            case 2:
                outputSavePath += inputFileName + "_" + modelName;
                break;
            case 3:
                outputSavePath += inputFileName + "_" + f.format(new Date());
                break;
            case 4:
                outputSavePath += inputFileName;
                break;
            default:
                outputSavePath += "output";
        }

        String cmd;
        if (inputIsGifAnimation) {
            outputSavePath += ".gif";
            cmd = ("cp " + dir + "/output.gif");
        } else if (format == 0) {
            outputSavePath += ".png";
            cmd = ("cp " + dir + "/output.png");
        } else {
            // 其他格式需要使用image magic进行转换，会额外消耗时间。但是为了方便，没有写到新线程上。
            // progress.setTitle(BUSY);
            if (format == 1) {
                outputSavePath += ".webp";
                cmd = ("./magick output.png");
            } else if (format == 2) {
                outputSavePath += ".gif";
                cmd = ("./magick output.png");
            } else if (format == 3) {
                outputSavePath += ".heic";
                cmd = ("./magick output.png");
            } else {
                outputSavePath += ".jpg";
                String q = formats[format].replaceAll("[a-zA-Z%\\s]+", "");
                if (q.length() > 0) {
                    cmd = ("./magick output.png -quality " + q);
                } else
                    cmd = ("./magick output.png");
            }
        }

        return cmd + " " + ShellUtils.escapeShellArgument(outputSavePath);
    }

}
