package com.tumuyan.ncnn.realsr;

/**
 * 图片格式识别工具：通过文件头（magic bytes）精准识别常见图片格式。
 *
 * 类型常量：
 *   TYPE_PNG    = 0
 *   TYPE_HEIF   = 1
 *   TYPE_GIF    = 2
 *   TYPE_AVIF   = 3
 *   TYPE_WEBP   = 4
 *   TYPE_JPG    = 5
 *   TYPE_BMP    = 6
 *   TYPE_UNKNOWN = -1
 */
public class PreprocessToPng {

    // 类型常量
    public static final int TYPE_PNG = 0;
    public static final int TYPE_HEIF = 1;
    public static final int TYPE_GIF = 2;
    public static final int TYPE_AVIF = 3;
    public static final int TYPE_WEBP = 4;
    public static final int TYPE_JPG = 5;
    public static final int TYPE_BMP = 6;
    public static final int TYPE_UNKNOWN = -1;

    // ---- 文件签名（magic bytes）----
    private static final byte[] PNG_SIG  = {(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}; // 8 字节
    private static final byte[] JPG_SIG  = {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF};                // 3 字节
    private static final byte[] BMP_SIG  = {'B', 'M'};                                             // 2 字节
    private static final byte[] GIF_SIG  = {'G', 'I', 'F', '8'};                                   // 4 字节，兼容 87a/89a
    private static final byte[] RIFF_SIG = {'R', 'I', 'F', 'F'};                                   // 4 字节
    private static final byte[] WEBP_SIG = {'W', 'E', 'B', 'P'};                                   // 4 字节，位于偏移 8
    private static final byte[] FTYP_SIG = {'f', 't', 'y', 'p'};                                   // 4 字节，位于偏移 4
    private static final byte[] AVIF_PREFIX = {'a', 'v', 'i'};                                     // major brand 前缀（avif/avis）
    private static final byte[] HEIF_PREFIX = {'h', 'e', 'i'};                                     // major brand 前缀（heic/heix/hevc...）

    /**
     * 识别图片格式。
     *
     * @param filehead 文件开头字节（建议至少 12 字节；越短可识别的格式越少）
     * @return TYPE_* 常量；无法识别时返回 TYPE_UNKNOWN
     */
    public static int match(byte[] filehead) {
        if (filehead == null || filehead.length < 4) return TYPE_UNKNOWN;

        // 1. PNG：0x89 'P' 'N' 'G' 0x0D 0x0A 0x1A 0x0A
        if (startsWith(filehead, PNG_SIG)) return TYPE_PNG;

        // 2. JPG：FF D8 FF
        if (startsWith(filehead, JPG_SIG)) return TYPE_JPG;

        // 3. BMP：'B' 'M'
        if (startsWith(filehead, BMP_SIG)) return TYPE_BMP;

        // 4. GIF：'G' 'I' 'F' '8'（兼容 GIF87a / GIF89a）
        if (startsWith(filehead, GIF_SIG)) return TYPE_GIF;

        // 5. WebP：RIFF + 文件大小 + WEBP
        if (startsWith(filehead, RIFF_SIG) && startsWithOffset(filehead, 8, WEBP_SIG))
            return TYPE_WEBP;

        // 6. ISO BMFF 容器（HEIF / HEIC / AVIF）：[4字节size] 'f' 't' 'y' 'p' [major brand]
        if (startsWithOffset(filehead, 4, FTYP_SIG)) {
            // major brand 前 3 字节判断：
            //   avi* -> AVIF（avif / avis）
            //   hei* -> HEIF（heic / heix / hevc / heim ...）
            //   其他（mif1 / msf1 / hvc1 ...）默认按 HEIF 处理
            if (startsWithOffset(filehead, 8, AVIF_PREFIX)) return TYPE_AVIF;
            if (startsWithOffset(filehead, 8, HEIF_PREFIX)) return TYPE_HEIF;
            return TYPE_HEIF;
        }

        return TYPE_UNKNOWN;
    }

    private static boolean startsWith(byte[] data, byte[] signature) {
        if (data == null || signature == null || data.length < signature.length) return false;
        for (int i = 0; i < signature.length; i++) {
            if (data[i] != signature[i]) return false;
        }
        return true;
    }

    private static boolean startsWithOffset(byte[] data, int offset, byte[] signature) {
        if (data == null || signature == null || offset < 0) return false;
        if (data.length < offset + signature.length) return false;
        for (int i = 0; i < signature.length; i++) {
            if (data[offset + i] != signature[i]) return false;
        }
        return true;
    }
}
