package com.tumuyan.ncnn.realsr;

/**
 * Shell命令安全处理工具类
 * 提供防止命令注入的参数转义功能
 */
public class ShellUtils {

    /**
     * 安全地转义单个Shell参数（核心方法）
     *
     * 使用单引号包裹 + 内部单引号转义技术：
     * - 单引号 '...' 内的所有字符都是字面意义
     * - 内部单引号通过 '\'' 模式转义（结束+转义+重新开始）
     *
     * 适用场景：
     * - 用户提供的文件路径（inputPath, outputPath, savePath等）
     * - 外部传入的配置值（extraPath, extraCommand等）
     * - 从URI解析的文件名
     *
     * 不适用场景（参数来源可信时）：
     * - 应用自身目录（getCacheDir(), getFilesDir()等）
     * - 硬编码的路径常量
     * - 已经验证过的内部变量
     *
     * 示例：
     * - "normal/path"          → 'normal/path'
     * - "path'with'quotes"    → 'path'\''with'\''quotes'
     * - "'; rm -rf /; '"      → ''\''; rm -rf /; '\''
     * - ""                     → '' (空字符串)
     * - null                   → '' (空字符串)
     *
     * @param arg 需要转义的原始字符串
     * @return 安全的Shell参数字符串，可直接拼接到命令中
     */
    public static String escapeShellArgument(String arg) {
        if (arg == null || arg.isEmpty()) {
            return "''";
        }

        StringBuilder sb = new StringBuilder("'");
        for (int i = 0; i < arg.length(); i++) {
            char c = arg.charAt(i);
            if (c == '\'') {
                sb.append("'\\''");  // 结束单引号 + 转义单引号 + 开始单引号
            } else {
                sb.append(c);
            }
        }
        sb.append("'");

        return sb.toString();
    }
}
