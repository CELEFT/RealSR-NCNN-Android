package com.tumuyan.ncnn.realsr;

public class PreprocessToPng {

    // 类型常量
    public static final int TYPE_PNG = 0;
    public static final int TYPE_HEIF = 1;
    public static final int TYPE_GIF = 2;
    public static final int TYPE_AVIF = 3;
    public static final int TYPE_WEBP = 4;
    // 其他格式返回 -1（JPG, BMP, 未知）

    private static final byte[] PNG_SIG = {(byte) 0X89, 0X50, 0X4E, 0X47, 0X0D, 0X0A, 0X1A, 0X0A};
    private static final byte[] JPG_SIG = {(byte) 0XFF, (byte) 0XD8};
    private static final byte[] WEBP_SIG = {0x52, 0x49, 0x46, 0x46};
    private static final byte[] BMP_SIG = {0x42, 0x4D};
    private static final byte[] HEIF_SIG = {0X00, 0X00, 0X00, 0X18, 0X66, 0X74, 0X79, 0X70, 0X68, 0X65, 0X69, 0X63, 0X00};
    private static final byte[] GIF_SIG = {0x47, 0x49, 0x46, 0x38};
    private static final byte[] AVIF_SIG = {0x61, 0x76, 0x69, 0x66};

    public static int match(byte[] filehead) {
        if (filehead == null || filehead.length < 12) return -1;

        // 检测 PNG
        if (startsWith(filehead, PNG_SIG)) return TYPE_PNG;
        // 检测 JPG（需转换）
        if (startsWith(filehead, JPG_SIG)) return -1;
        // 检测 WebP
        if (startsWith(filehead, WEBP_SIG)) return TYPE_WEBP;
        // 检测 BMP（需转换）
        if (startsWith(filehead, BMP_SIG)) return -1;
        // 检测 HEIF
        if (startsWith(filehead, HEIF_SIG)) return TYPE_HEIF;
        // 检测 GIF
        if (startsWith(filehead, GIF_SIG)) return TYPE_GIF;
        // 检测 AVIF（偏移8字节）
        if (startsWithOffset(filehead, 8, AVIF_SIG)) return TYPE_AVIF;

        return -1; // 未知格式，交给 ImageMagick 尝试
    }

    private static boolean startsWith(byte[] data, byte[] signature) {
        if (data.length < signature.length) return false;
        for (int i = 0; i < signature.length; i++) {
            if (data[i] != signature[i]) return false;
        }
        return true;
    }

    private static boolean startsWithOffset(byte[] data, int offset, byte[] signature) {
        if (data.length < offset + signature.length) return false;
        for (int i = 0; i < signature.length; i++) {
            if (data[offset + i] != signature[i]) return false;
        }
        return true;
    }
}