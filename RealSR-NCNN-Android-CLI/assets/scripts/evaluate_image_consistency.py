import os
import sys
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from scipy.stats import entropy


class ImageAnomalyDetector:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.results = []
        
    def load_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        if img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        
        return rgb, alpha
    
    def resize_image(self, img, target_size):
        if len(img.shape) == 2:
            return cv2.resize(img, (target_size[1], target_size[0]), 
                            interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img, (target_size[1], target_size[0]), 
                            interpolation=cv2.INTER_AREA)
    
    def detect_alpha_anomalies(self, input_alpha, output_alpha):
        h, w = output_alpha.shape
        input_alpha_resized = self.resize_image(input_alpha, (h, w))
        
        anomalies = {}
        
        alpha_diff = np.abs(output_alpha - input_alpha_resized)
        
        opaque_mask = input_alpha_resized > 0.8
        transparent_mask = output_alpha < 0.3
        sudden_transparency = opaque_mask & transparent_mask
        
        sudden_transparency_ratio = np.sum(sudden_transparency) / (h * w)
        anomalies['sudden_transparency_ratio'] = sudden_transparency_ratio
        
        if sudden_transparency_ratio > 0.05:
            anomalies['sudden_transparency_severity'] = 'severe'
        elif sudden_transparency_ratio > 0.01:
            anomalies['sudden_transparency_severity'] = 'moderate'
        else:
            anomalies['sudden_transparency_severity'] = 'none'
        
        input_edges = cv2.Canny((input_alpha_resized * 255).astype(np.uint8), 100, 200)
        output_edges = cv2.Canny((output_alpha * 255).astype(np.uint8), 100, 200)
        
        edge_diff = np.abs(input_edges.astype(np.float32) - output_edges.astype(np.float32))
        edge_change_ratio = np.sum(edge_diff > 50) / (h * w)
        anomalies['edge_change_ratio'] = edge_change_ratio
        
        input_hist, _ = np.histogram(input_alpha_resized, bins=50, range=(0, 1), density=True)
        output_hist, _ = np.histogram(output_alpha, bins=50, range=(0, 1), density=True)
        
        hist_diff = np.abs(input_hist - output_hist)
        hist_divergence = np.sum(hist_diff)
        anomalies['histogram_divergence'] = hist_divergence
        
        input_entropy = entropy(input_hist + 1e-10)
        output_entropy = entropy(output_hist + 1e-10)
        entropy_change = abs(input_entropy - output_entropy)
        anomalies['entropy_change'] = entropy_change
        
        alpha_anomaly_score = (
            sudden_transparency_ratio * 1000 +
            edge_change_ratio * 500 +
            hist_divergence * 10 +
            entropy_change * 20
        )
        anomalies['alpha_anomaly_score'] = min(alpha_anomaly_score, 100)
        
        return anomalies
    
    def detect_rgb_anomalies(self, input_rgb, output_rgb, input_alpha, output_alpha):
        h, w = output_rgb.shape[:2]
        input_rgb_resized = self.resize_image(input_rgb, (h, w))
        
        anomalies = {}
        
        for i, channel in enumerate(['R', 'G', 'B']):
            input_ch = input_rgb_resized[:, :, i]
            output_ch = output_rgb[:, :, i]
            
            mean_diff = abs(np.mean(input_ch) - np.mean(output_ch))
            std_diff = abs(np.std(input_ch) - np.std(output_ch))
            
            anomalies[f'{channel}_mean_diff'] = mean_diff
            anomalies[f'{channel}_std_diff'] = std_diff
        
        input_mean = np.mean(input_rgb_resized, axis=(0, 1))
        output_mean = np.mean(output_rgb, axis=(0, 1))
        
        input_ratio = input_mean / (np.sum(input_mean) + 1e-10)
        output_ratio = output_mean / (np.sum(output_mean) + 1e-10)
        
        ratio_diff = np.abs(input_ratio - output_ratio)
        max_ratio_diff = np.max(ratio_diff)
        anomalies['max_channel_ratio_diff'] = max_ratio_diff
        
        if max_ratio_diff > 0.3:
            anomalies['channel_ratio_severity'] = 'severe'
        elif max_ratio_diff > 0.15:
            anomalies['channel_ratio_severity'] = 'moderate'
        else:
            anomalies['channel_ratio_severity'] = 'none'
        
        input_gray = cv2.cvtColor(input_rgb_resized, cv2.COLOR_RGB2GRAY)
        output_gray = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2GRAY)
        
        ssim_score = ssim(input_gray, output_gray, data_range=255)
        anomalies['ssim_score'] = ssim_score
        
        if ssim_score < 0.7:
            anomalies['ssim_severity'] = 'severe'
        elif ssim_score < 0.85:
            anomalies['ssim_severity'] = 'moderate'
        else:
            anomalies['ssim_severity'] = 'none'
        
        mse = np.mean((input_rgb_resized.astype(np.float32) - output_rgb.astype(np.float32)) ** 2)
        anomalies['mse'] = mse
        
        if mse > 0:
            psnr = 10 * np.log10(255 ** 2 / mse)
            anomalies['psnr'] = psnr
        else:
            anomalies['psnr'] = 100
        
        input_alpha_resized = self.resize_image(input_alpha, (h, w))
        opaque_mask = (input_alpha_resized > 0.5) & (output_alpha > 0.5)
        
        if np.sum(opaque_mask) > 0:
            input_opaque = input_rgb_resized[opaque_mask]
            output_opaque = output_rgb[opaque_mask]
            
            if len(input_opaque) > 0:
                input_opaque_gray = cv2.cvtColor(input_opaque.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()
                output_opaque_gray = cv2.cvtColor(output_opaque.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()
                
                if len(input_opaque_gray) > 10:
                    ssim_opaque = ssim(
                        input_opaque_gray.reshape(-1), 
                        output_opaque_gray.reshape(-1), 
                        data_range=255
                    )
                    anomalies['ssim_opaque'] = ssim_opaque
        
        laplacian_input = cv2.Laplacian(input_gray, cv2.CV_64F)
        laplacian_output = cv2.Laplacian(output_gray, cv2.CV_64F)
        
        noise_input = np.std(laplacian_input)
        noise_output = np.std(laplacian_output)
        noise_ratio = noise_output / (noise_input + 1e-10)
        anomalies['noise_ratio'] = noise_ratio
        
        if noise_ratio > 2.0:
            anomalies['noise_severity'] = 'severe'
        elif noise_ratio > 1.5:
            anomalies['noise_severity'] = 'moderate'
        else:
            anomalies['noise_severity'] = 'none'
        
        rgb_anomaly_score = (
            max_ratio_diff * 100 +
            (1 - ssim_score) * 50 +
            min(mse / 1000, 1) * 20 +
            abs(1 - noise_ratio) * 30
        )
        anomalies['rgb_anomaly_score'] = min(rgb_anomaly_score, 100)
        
        return anomalies
    
    def calculate_overall_score(self, alpha_anomalies, rgb_anomalies):
        alpha_score = alpha_anomalies.get('alpha_anomaly_score', 0)
        rgb_score = rgb_anomalies.get('rgb_anomaly_score', 0)
        
        overall_score = alpha_score * 0.6 + rgb_score * 0.4
        
        if overall_score > 60:
            severity = 'severe'
        elif overall_score > 30:
            severity = 'moderate'
        elif overall_score > 10:
            severity = 'mild'
        else:
            severity = 'normal'
        
        return {
            'overall_score': overall_score,
            'severity': severity
        }
    
    def parse_output_filename(self, filename):
        stem = filename.stem
        
        if '_' not in stem:
            return '', '', stem
        
        parts = stem.split('_')
        if len(parts) < 2:
            return '', '', stem
        
        input_name = parts[-1]
        program = parts[0]
        params = '_'.join(parts[1:-1]) if len(parts) > 2 else ''
        
        return program, params, input_name
    
    def extract_inference_program(self, program):
        return program if program else 'unknown'
    
    def match_files(self):
        input_files = list(self.input_dir.glob('*.png')) + list(self.input_dir.glob('*.jpg'))
        output_files = list(self.output_dir.glob('*.png')) + list(self.output_dir.glob('*.jpg'))
        
        matches = []
        for output_file in output_files:
            program, params, input_name = self.parse_output_filename(output_file)
            input_file = self.input_dir / f"{input_name}{output_file.suffix}"
            
            if input_file.exists():
                matches.append({
                    'input_file': input_file,
                    'output_file': output_file,
                    'program': program,
                    'params': params,
                    'input_name': input_name
                })
        
        return matches
    
    def evaluate(self):
        matches = self.match_files()
        
        for match in matches:
            try:
                input_rgb, input_alpha = self.load_image(match['input_file'])
                output_rgb, output_alpha = self.load_image(match['output_file'])
                
                alpha_anomalies = self.detect_alpha_anomalies(input_alpha, output_alpha)
                rgb_anomalies = self.detect_rgb_anomalies(input_rgb, output_rgb, input_alpha, output_alpha)
                overall = self.calculate_overall_score(alpha_anomalies, rgb_anomalies)
                
                result = {
                    'output_filename': match['output_file'].name,
                    'input_filename': match['input_file'].name,
                    'params': match['params'],
                    'program': match['program'],
                    'input_name': match['input_name'],
                    'output_size': output_rgb.shape[:2],
                    'input_size': input_rgb.shape[:2],
                    'alpha_anomalies': alpha_anomalies,
                    'rgb_anomalies': rgb_anomalies,
                    'overall': overall
                }
                
                self.results.append(result)
                print(f"Evaluated: {match['output_file'].name} - Severity: {overall['severity']}")
                
            except Exception as e:
                print(f"Error processing {match['output_file'].name}: {e}")
                continue
        
        return self.results
    
    def generate_html_report(self, output_path, csv_data=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"evaluation_report_{timestamp}.html"
        
        if csv_data and 'csv_file' in csv_data and csv_data['csv_file']:
            csv_path = Path(csv_data['csv_file'])
            if csv_path.exists():
                report_filename = csv_path.stem + '.html'
        
        report_path = Path(output_path) / report_filename
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        severity_counts = {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 0}
        for result in self.results:
            severity_counts[result['overall']['severity']] += 1
        
        html_content = self._generate_html_content(severity_counts, csv_data)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_path}")
        return report_path
    
    def _generate_html_content(self, severity_counts, csv_data=None):
        total = len(self.results)
        
        csv_stats = None
        if csv_data:
            csv_keys = set()
            csv_empty_outputs = 0
            for key, value in csv_data.items():
                if isinstance(value, dict):
                    csv_keys.add(value['output_filename'])
                    if not value['output_filename']:
                        csv_empty_outputs += 1
            
            matched = 0
            unmatched = 0
            empty_output = 0
            
            for result in self.results:
                if result['output_filename'] in csv_keys:
                    matched += 1
                else:
                    unmatched += 1
            
            csv_stats = {
                'total_csv': len(csv_data) - 1,
                'matched': matched,
                'unmatched': unmatched,
                'empty_output': csv_empty_outputs
            }
        
        rows = []
        for i, result in enumerate(self.results):
            row_class = f"severity-{result['overall']['severity']}"
            image_path = f"../output/{result['output_filename']}"
            
            row = f"""
            <tr class="{row_class}" data-severity="{result['overall']['severity']}" data-program="{result['program']}" data-input="{result['input_filename']}" onclick="showImage('{image_path}')">
                <td data-value="{i + 1}">{i + 1}</td>
                <td data-value="{result['output_filename']}">{result['output_filename']}</td>
                <td data-value="{result['input_filename']}">{result['input_filename']}</td>
                <td data-value="{result['program']}">{result['program']}</td>
                <td data-value="{result['params']}">{result['params']}</td>
                <td data-value="{result['output_size'][0] * result['output_size'][1]}">{result['output_size'][0]}x{result['output_size'][1]}</td>
                <td data-value="{result['input_size'][0] * result['input_size'][1]}">{result['input_size'][0]}x{result['input_size'][1]}</td>
                <td data-value="{result['alpha_anomalies']['sudden_transparency_severity']}">{result['alpha_anomalies']['sudden_transparency_severity']}</td>
                <td data-value="{result['alpha_anomalies']['sudden_transparency_ratio']:.4f}">{result['alpha_anomalies']['sudden_transparency_ratio']:.4f}</td>
                <td data-value="{result['alpha_anomalies']['alpha_anomaly_score']:.2f}">{result['alpha_anomalies']['alpha_anomaly_score']:.2f}</td>
                <td data-value="{result['rgb_anomalies']['ssim_score']:.4f}">{result['rgb_anomalies']['ssim_score']:.4f}</td>
                <td data-value="{result['rgb_anomalies']['psnr']:.2f}">{result['rgb_anomalies']['psnr']:.2f}</td>
                <td data-value="{result['rgb_anomalies']['channel_ratio_severity']}">{result['rgb_anomalies']['channel_ratio_severity']}</td>
                <td data-value="{result['rgb_anomalies']['rgb_anomaly_score']:.2f}">{result['rgb_anomalies']['rgb_anomaly_score']:.2f}</td>
                <td data-value="{result['overall']['overall_score']:.2f}">{result['overall']['overall_score']:.2f}</td>
                <td class="severity-{result['overall']['severity']}" data-value="{result['overall']['severity']}">{result['overall']['severity']}</td>
            </tr>
            """
            rows.append(row)
        
        if csv_data:
            csv_keys = set()
            for key, value in csv_data.items():
                if isinstance(value, dict):
                    csv_keys.add(f"{value['input_filename']}_{value['program_name']}_{value['param_group']}")
            
            for key, value in csv_data.items():
                if isinstance(value, dict):
                    csv_key = f"{value['input_filename']}_{value['program_name']}_{value['param_group']}"
                    if csv_key not in csv_keys:
                        matched += 1
                        if not value['output_filename']:
                            empty_output += 1
                        
                        row_class = "severity-none"
                        row = f"""
                        <tr class="{row_class}" data-severity="none" data-program="{value['program_name']}" data-input="{value['input_filename']}">
                            <td data-value="{len(rows) + 1}">{len(rows) + 1}</td>
                            <td data-value="{value['output_filename']}">{value['output_filename']}</td>
                            <td data-value="{value['input_filename']}">{value['input_filename']}</td>
                            <td data-value="{value['program_name']}">{value['program_name']}</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td data-value="">-</td>
                            <td class="severity-none" data-value="none">无输出</td>
                        </tr>
                        """
                        rows.append(row)
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图像异常检测报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .timestamp {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .summary {{
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .summary h2 {{
            color: #495057;
            margin-bottom: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        }}
        
        .stat-card.active {{
            ring: 4px solid #667eea;
            transform: scale(1.05);
        }}
        
        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-card .label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .stat-card.normal .number {{ color: #28a745; }}
        .stat-card.mild .number {{ color: #ffc107; }}
        .stat-card.moderate .number {{ color: #fd7e14; }}
        .stat-card.severe .number {{ color: #dc3545; }}
        
        .stat-card.severity-none .number {{ color: #6c757d; }}
        
        .csv-stats {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .csv-stats h3 {{
            color: #495057;
            margin-bottom: 10px;
        }}
        
        .csv-stats div {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .csv-stats div > div {{
            flex: 1;
            min-width: 200px;
        }}
        
        .table-container {{
            padding: 30px;
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        
        thead {{
            background: #495057;
            color: white;
        }}
        
        th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
            position: relative;
        }}
        
        th:hover {{
            background: #5a6268;
        }}
        
        th::after {{
            content: '↕';
            position: absolute;
            right: 8px;
            opacity: 0.3;
            font-size: 0.8em;
        }}
        
        th.asc::after {{
            content: '↑';
            opacity: 1;
        }}
        
        th.desc::after {{
            content: '↓';
            opacity: 1;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        tbody tr {{
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        tbody tr:hover {{
            background: #f8f9fa;
            transform: scale(1.01);
        }}
        
        tbody tr.severity-normal {{
            background: rgba(40, 167, 69, 0.1);
        }}
        
        tbody tr.severity-mild {{
            background: rgba(255, 193, 7, 0.1);
        }}
        
        tbody tr.severity-moderate {{
            background: rgba(253, 126, 20, 0.1);
        }}
        
        tbody tr.severity-severe {{
            background: rgba(220, 53, 69, 0.1);
        }}
        
        td.severity-normal {{
            color: #28a745;
            font-weight: bold;
        }}
        
        td.severity-mild {{
            color: #ffc107;
            font-weight: bold;
        }}
        
        td.severity-moderate {{
            color: #fd7e14;
            font-weight: bold;
        }}
        
        td.severity-severe {{
            color: #dc3545;
            font-weight: bold;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
        }}
        
        .modal-content {{
            position: relative;
            margin: 2% auto;
            max-width: 95%;
            max-height: 95%;
            background: white;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .modal-header {{
            background: #495057;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .close {{
            color: white;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.2s;
        }}
        
        .close:hover {{
            color: #ff6b6b;
        }}
        
        .modal-body {{
            padding: 40px;
            overflow: auto;
            max-height: calc(95vh - 80px);
            position: relative;
            background: #1a1a1a;
            overflow-x: auto;
            overflow-y: auto;
        }}
        
        .modal-body img {{
            max-width: none;
            height: auto;
            display: block;
            margin: 0;
            transition: transform 0.1s ease-out;
            cursor: grab;
            user-select: none;
        }}
        
        .modal-body.bg-black {{
            background: #000;
        }}
        
        .modal-body.bg-white {{
            background: #fff;
        }}
        
        .modal-body.bg-gray {{
            background: #808080;
        }}
        
        .modal-body.bg-red {{
            background: #ff0000;
        }}
        
        .modal-body.bg-green {{
            background: #00ff00;
        }}
        
        .modal-body.bg-blue {{
            background: #0000ff;
        }}
        
        .magnifier {{
            position: absolute;
            border: 3px solid #fff;
            border-radius: 4px;
            cursor: none;
            width: 150px;
            height: 150px;
            pointer-events: none;
            display: none;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
            z-index: 10;
        }}
        
        .hand-tool {{
            position: fixed;
            bottom: 100px;
            left: 30px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 1002;
        }}
        
        .hand-btn {{
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: #667eea;
            color: white;
            border: none;
            font-size: 24px;
            cursor: grab;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .hand-btn:hover {{
            background: #764ba2;
            transform: scale(1.1);
        }}
        
        .hand-btn.active {{
            background: #764ba2;
            box-shadow: 0 0 10px rgba(118, 75, 162, 0.5);
        }}
        
        .hand-btn.grabbing {{
            cursor: grabbing;
        }}
        
        .zoom-controls {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 1001;
        }}
        
        .zoom-btn {{
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: #667eea;
            color: white;
            border: none;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.2s;
        }}
        
        .zoom-btn:hover {{
            background: #764ba2;
            transform: scale(1.1);
        }}
        
        .zoom-btn.active {{
            background: #764ba2;
            box-shadow: 0 0 10px rgba(118, 75, 162, 0.5);
        }}
        
        .zoom-slider-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .zoom-slider {{
            width: 150px;
            cursor: pointer;
        }}
        
        .zoom-value {{
            font-size: 12px;
            font-weight: bold;
            color: #495057;
        }}
        
        .bg-controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        
        .bg-btn {{
            width: 30px;
            height: 30px;
            border: 2px solid #fff;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .bg-btn:hover {{
            transform: scale(1.2);
        }}
        
        .bg-btn.active {{
            border-color: #667eea;
            box-shadow: 0 0 8px rgba(102, 126, 234, 0.5);
        }}
        
        .bg-btn.bg-black {{ background: #000; }}
        .bg-btn.bg-white {{ background: #fff; }}
        .bg-btn.bg-gray {{ background: #808080; }}
        .bg-btn.bg-red {{ background: #ff0000; }}
        .bg-btn.bg-green {{ background: #00ff00; }}
        .bg-btn.bg-blue {{ background: #0000ff; }}
        
        .magnifier-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-left: 20px;
        }}
        
        .toggle-switch {{
            position: relative;
            width: 50px;
            height: 26px;
        }}
        
        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        
        .slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 26px;
        }}
        
        .slider:before {{
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }}
        
        input:checked + .slider {{
            background-color: #667eea;
        }}
        
        input:checked + .slider:before {{
            transform: translateX(24px);
        }}
        
        .legend {{
            padding: 20px 30px;
            background: #f8f9fa;
            border-top: 2px solid #e9ecef;
        }}
        
        .legend h3 {{
            color: #495057;
            margin-bottom: 15px;
        }}
        
        .legend-items {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>图像异常检测报告</h1>
            <div class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="summary">
            <h2>统计摘要 (点击筛选)</h2>
            <div class="stats-grid">
                <div class="stat-card" onclick="filterBySeverity('all')">
                    <div class="number">{total}</div>
                    <div class="label">总测试数</div>
                </div>
                <div class="stat-card normal" onclick="filterBySeverity('normal')">
                    <div class="number">{severity_counts['normal']}</div>
                    <div class="label">正常</div>
                </div>
                <div class="stat-card mild" onclick="filterBySeverity('mild')">
                    <div class="number">{severity_counts['mild']}</div>
                    <div class="label">轻微异常</div>
                </div>
                <div class="stat-card moderate" onclick="filterBySeverity('moderate')">
                    <div class="number">{severity_counts['moderate']}</div>
                    <div class="label">中度异常</div>
                </div>
                <div class="stat-card severe" onclick="filterBySeverity('severe')">
                    <div class="number">{severity_counts['severe']}</div>
                    <div class="label">严重异常</div>
                </div>
            </div>
            
            {f'<div class="csv-stats" style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">' if csv_data else ''}
            <h3 style="margin-bottom: 10px; color: #495057;">CSV文件统计信息</h3>
            {f'<div style="display: flex; gap: 20px; flex-wrap: wrap;">' if csv_data else ''}
            {f'<div style="flex: 1; min-width: 200px;"><strong>CSV文件:</strong> {csv_data.get("csv_file", "无")}</div>' if csv_data else ''}
            {f'<div style="flex: 1; min-width: 200px;"><strong>总记录数:</strong> {csv_stats["total_csv"] if csv_stats else 0}</div>' if csv_data else ''}
            {f'<div style="flex: 1; min-width: 200px;"><strong>匹配记录数:</strong> {csv_stats["matched"] if csv_stats else 0}</div>' if csv_data else ''}
            {f'<div style="flex: 1; min-width: 200px;"><strong>未匹配记录数:</strong> {csv_stats["unmatched"] if csv_stats else 0}</div>' if csv_data else ''}
            {f'<div style="flex: 1; min-width: 200px;"><strong>空输出记录数:</strong> {csv_stats["empty_output"] if csv_stats else 0}</div>' if csv_data else ''}
            {f'</div>' if csv_data else ''}
            {f'</div>' if csv_data else ''}
            
            <div class="filter-controls" style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <h3 style="margin-bottom: 10px; color: #495057;">高级筛选</h3>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <label for="programFilter" style="font-weight: 600;">推理程序:</label>
                        <select id="programFilter" onchange="applyFilters()" style="padding: 8px; border-radius: 4px; border: 1px solid #ced4da; min-width: 150px;">
                            <option value="all">全部</option>
                        </select>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <label for="inputFilter" style="font-weight: 600;">输入图片:</label>
                        <select id="inputFilter" onchange="applyFilters()" style="padding: 8px; border-radius: 4px; border: 1px solid #ced4da; min-width: 150px;">
                            <option value="all">全部</option>
                        </select>
                    </div>
                    <button onclick="resetFilters()" style="padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">重置筛选</button>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>输出文件</th>
                        <th>输入文件</th>
                        <th>推理程序</th>
                        <th>参数</th>
                        <th>输出尺寸</th>
                        <th>输入尺寸</th>
                        <th>Alpha突变</th>
                        <th>突变比例</th>
                        <th>Alpha评分</th>
                        <th>SSIM</th>
                        <th>PSNR</th>
                        <th>通道异常</th>
                        <th>RGB评分</th>
                        <th>总评分</th>
                        <th>严重程度</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        
        <div class="legend">
            <h3>图例说明</h3>
            <div class="legend-items">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(40, 167, 69, 0.3);"></div>
                    <span>正常 (评分 ≤ 10)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(255, 193, 7, 0.3);"></div>
                    <span>轻微异常 (10 < 评分 ≤ 30)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(253, 126, 20, 0.3);"></div>
                    <span>中度异常 (30 < 评分 ≤ 60)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(220, 53, 69, 0.3);"></div>
                    <span>严重异常 (评分 > 60)</span>
                </div>
            </div>
        </div>
    </div>
    
    <div id="imageModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <span id="modalTitle">图像详情</span>
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div class="zoom-slider-container">
                        <div class="zoom-value">缩放: <span id="zoomValue">100</span>%</div>
                        <input type="range" id="zoomSlider" class="zoom-slider" min="20" max="500" value="100" oninput="updateZoomFromSlider()">
                    </div>
                    <div class="bg-controls">
                        <div class="bg-btn bg-black" onclick="changeBackground('black')" title="黑色背景"></div>
                        <div class="bg-btn bg-white" onclick="changeBackground('white')" title="白色背景"></div>
                        <div class="bg-btn bg-gray active" onclick="changeBackground('gray')" title="灰色背景"></div>
                        <div class="bg-btn bg-red" onclick="changeBackground('red')" title="红色背景"></div>
                        <div class="bg-btn bg-green" onclick="changeBackground('green')" title="绿色背景"></div>
                        <div class="bg-btn bg-blue" onclick="changeBackground('blue')" title="蓝色背景"></div>
                    </div>
                    <div class="magnifier-toggle">
                        <span>放大镜</span>
                        <label class="toggle-switch">
                            <input type="checkbox" id="magnifierToggle" checked onchange="toggleMagnifier()">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <span class="close" onclick="closeModal()">&times;</span>
                </div>
            </div>
            <div class="modal-body bg-gray" id="modalBody">
                <img id="modalImage" src="" alt="Output Image">
                <div class="magnifier" id="magnifier"></div>
            </div>
            
            <div class="hand-tool" id="handTool">
                <button class="hand-btn" id="handBtn" onclick="toggleHandTool()" title="抓手工具">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 11V6a2 2 2 0 0-2-2h-2a2 2 0 0 2 2v12a2 2 2 0 0 2 2H6a2 2 0 0-2 2 2-2 2-2V6a2 2 2 0 0 2 2h12a2 2 2 0 0 2 2v-2a2 2 2 0 0 2-2H6z"/>
                    </svg>
                </button>
            </div>
        </div>
    </div>
    
    <script>
        let currentZoom = 1;
        let currentFilter = 'all';
        let currentSortColumn = -1;
        let sortDirection = 'asc';
        let magnifierEnabled = true;
        let magnifierZoom = 2;
        let currentBackground = 'gray';
        let handToolEnabled = false;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let imageOffsetX = 0;
        let imageOffsetY = 0;
        
        document.addEventListener('DOMContentLoaded', function() {{
            const table = document.querySelector('table');
            const headers = table.querySelectorAll('th');
            
            headers.forEach((header, index) => {{
                header.addEventListener('click', () => {{
                    sortTable(index);
                }});
            }});
            
            const modalBody = document.getElementById('modalBody');
            const modalImage = document.getElementById('modalImage');
            const magnifier = document.getElementById('magnifier');
            
            modalBody.addEventListener('mousemove', function(e) {{
                if (!magnifierEnabled) return;
                
                const rect = modalImage.getBoundingClientRect();
                const bodyRect = modalBody.getBoundingClientRect();
                
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {{
                    magnifier.style.display = 'block';
                    
                    const magnifierX = e.clientX - bodyRect.left - 75;
                    const magnifierY = e.clientY - bodyRect.top - 75;
                    
                    magnifier.style.left = magnifierX + 'px';
                    magnifier.style.top = magnifierY + 'px';
                    
                    const bgX = -(x * magnifierZoom - 75);
                    const bgY = -(y * magnifierZoom - 75);
                    
                    magnifier.style.backgroundImage = `url(${{modalImage.src}})`;
                    magnifier.style.backgroundSize = `${{rect.width * magnifierZoom}}px ${{rect.height * magnifierZoom}}px`;
                    magnifier.style.backgroundPosition = `${{bgX}}px ${{bgY}}px`;
                }} else {{
                    magnifier.style.display = 'none';
                }}
            }});
            
            modalBody.addEventListener('mouseleave', function() {{
                magnifier.style.display = 'none';
            }});
            
            const img = document.getElementById('modalImage');
            
            img.addEventListener('mousedown', function(e) {{
                if (!handToolEnabled) return;
                isDragging = true;
                dragStartX = e.clientX - imageOffsetX;
                dragStartY = e.clientY - imageOffsetY;
                img.style.cursor = 'grabbing';
            }});
            
            document.addEventListener('mousemove', function(e) {{
                if (!isDragging || !handToolEnabled) return;
                e.preventDefault();
                const newX = e.clientX - dragStartX;
                const newY = e.clientY - dragStartY;
                imageOffsetX = newX;
                imageOffsetY = newY;
                img.style.transform = `translate(${{newX}}px, ${{newY}}px)`;
            }});
            
            document.addEventListener('mouseup', function() {{
                isDragging = false;
                if (handToolEnabled) {{
                    img.style.cursor = 'grab';
                }}
            }});
            
            document.addEventListener('mouseleave', function(e) {{
                if (e.target === img && isDragging) {{
                    isDragging = false;
                    if (handToolEnabled) {{
                        img.style.cursor = 'grab';
                    }}
                }}
            }});
            
            populateFilters();
        }});
        
        function sortTable(columnIndex) {{
            const table = document.querySelector('table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            if (currentSortColumn === columnIndex) {{
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            }} else {{
                currentSortColumn = columnIndex;
                sortDirection = 'asc';
            }}
            
            const headers = table.querySelectorAll('th');
            headers.forEach((header, index) => {{
                header.classList.remove('asc', 'desc');
                if (index === columnIndex) {{
                    header.classList.add(sortDirection);
                }}
            }});
            
            rows.sort((a, b) => {{
                const aValue = a.cells[columnIndex].getAttribute('data-value');
                const bValue = b.cells[columnIndex].getAttribute('data-value');
                
                let comparison = 0;
                
                if (!isNaN(parseFloat(aValue)) && !isNaN(parseFloat(bValue))) {{
                    comparison = parseFloat(aValue) - parseFloat(bValue);
                }} else {{
                    comparison = aValue.localeCompare(bValue);
                }}
                
                return sortDirection === 'asc' ? comparison : -comparison;
            }});
            
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        function filterBySeverity(severity) {{
            currentFilter = severity;
            applyFilters();
        }}
        
        function populateFilters() {{
            const rows = document.querySelectorAll('tbody tr');
            const programs = new Set();
            const inputs = new Set();
            
            rows.forEach(row => {{
                const program = row.getAttribute('data-program');
                const input = row.getAttribute('data-input');
                if (program) programs.add(program);
                if (input) inputs.add(input);
            }});
            
            const programFilter = document.getElementById('programFilter');
            const inputFilter = document.getElementById('inputFilter');
            
            programs.forEach(program => {{
                const option = document.createElement('option');
                option.value = program;
                option.textContent = program;
                programFilter.appendChild(option);
            }});
            
            inputs.forEach(input => {{
                const option = document.createElement('option');
                option.value = input;
                option.textContent = input;
                inputFilter.appendChild(option);
            }});
        }}
        
        function applyFilters() {{
            const severityFilter = currentFilter;
            const programFilter = document.getElementById('programFilter').value;
            const inputFilter = document.getElementById('inputFilter').value;
            
            const rows = document.querySelectorAll('tbody tr');
            const cards = document.querySelectorAll('.stat-card');
            
            cards.forEach(card => card.classList.remove('active'));
            
            if (severityFilter === 'all') {{
                cards[0].classList.add('active');
            }} else {{
                const severityMap = {{'normal': 1, 'mild': 2, 'moderate': 3, 'severe': 4}};
                cards[severityMap[severityFilter]].classList.add('active');
            }}
            
            rows.forEach(row => {{
                const severity = row.getAttribute('data-severity');
                const program = row.getAttribute('data-program');
                const input = row.getAttribute('data-input');
                
                const severityMatch = severityFilter === 'all' || severity === severityFilter;
                const programMatch = programFilter === 'all' || program === programFilter;
                const inputMatch = inputFilter === 'all' || input === inputFilter;
                
                if (severityMatch && programMatch && inputMatch) {{
                    row.style.display = '';
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}
        
        function resetFilters() {{
            currentFilter = 'all';
            document.getElementById('programFilter').value = 'all';
            document.getElementById('inputFilter').value = 'all';
            applyFilters();
        }}
        
        function showImage(imagePath) {{
            const modal = document.getElementById('imageModal');
            const img = document.getElementById('modalImage');
            const title = document.getElementById('modalTitle');
            
            img.src = imagePath;
            title.textContent = imagePath.split('/').pop();
            modal.style.display = 'block';
            currentZoom = 1;
            updateZoom();
            updateZoomSlider();
            
            changeBackground('gray');
            document.getElementById('magnifierToggle').checked = true;
            magnifierEnabled = true;
            
            imageOffsetX = 0;
            imageOffsetY = 0;
            img.style.transform = 'translate(0px, 0px)';
        }}
        
        function toggleHandTool() {{
            handToolEnabled = !handToolEnabled;
            const btn = document.getElementById('handBtn');
            const img = document.getElementById('modalImage');
            
            if (handToolEnabled) {{
                btn.classList.add('active');
                btn.classList.add('grabbing');
                img.style.cursor = 'grab';
            }} else {{
                btn.classList.remove('active');
                btn.classList.remove('grabbing');
                img.style.cursor = 'default';
            }}
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
            document.getElementById('magnifierToggle').checked = false;
            magnifierEnabled = false;
            document.getElementById('magnifier').style.display = 'none';
            
            handToolEnabled = false;
            const btn = document.getElementById('handBtn');
            if (btn) {{
                btn.classList.remove('active');
                btn.classList.remove('grabbing');
            }}
            
            const img = document.getElementById('modalImage');
            img.style.cursor = 'default';
            img.style.transform = 'translate(0px, 0px)';
            imageOffsetX = 0;
            imageOffsetY = 0;
        }}
        
        function zoomIn() {{
            currentZoom = Math.min(currentZoom * 1.2, 5);
            updateZoom();
            updateZoomSlider();
        }}
        
        function zoomOut() {{
            currentZoom = Math.max(currentZoom / 1.2, 0.2);
            updateZoom();
            updateZoomSlider();
        }}
        
        function resetZoom() {{
            currentZoom = 1;
            imageOffsetX = 0;
            imageOffsetY = 0;
            updateZoom();
            updateZoomSlider();
        }}
        
        function updateZoomSlider() {{
            const slider = document.getElementById('zoomSlider');
            const value = Math.round(currentZoom * 100);
            slider.value = value;
            document.getElementById('zoomValue').textContent = value;
        }}
        
        function updateZoom() {{
            const img = document.getElementById('modalImage');
            img.style.transform = `scale(${{currentZoom}}) translate(${{imageOffsetX}}px, ${{imageOffsetY}}px)`;
            img.style.transformOrigin = 'center center';
        }}
        
        function updateZoomFromSlider() {{
            const slider = document.getElementById('zoomSlider');
            const value = slider.value;
            currentZoom = value / 100;
            document.getElementById('zoomValue').textContent = value;
            updateZoom();
        }}
        
        function changeBackground(color) {{
            const modalBody = document.getElementById('modalBody');
            const bgButtons = document.querySelectorAll('.bg-btn');
            
            bgButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelector(`.bg-btn.bg-${{color}}`).classList.add('active');
            
            modalBody.className = `modal-body bg-${{color}}`;
        }}
        
        function toggleMagnifier() {{
            magnifierEnabled = document.getElementById('magnifierToggle').checked;
            if (!magnifierEnabled) {{
                document.getElementById('magnifier').style.display = 'none';
            }}
        }}
        
        window.onclick = function(event) {{
            const modal = document.getElementById('imageModal');
            if (event.target == modal) {{
                closeModal();
            }}
        }}
        
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                closeModal();
            }} else if (e.key === '+' || e.key === '=') {{
                zoomIn();
            }} else if (e.key === '-') {{
                zoomOut();
            }} else if (e.key === '0') {{
                resetZoom();
            }}
        }});
        
        filterBySeverity('all');
    </script>
</body>
</html>
        """
        
        return html


def main():
    script_dir = Path(__file__).parent
    assets_dir = script_dir.parent
    input_dir = assets_dir / 'input'
    output_dir = assets_dir / 'output'
    report_dir = assets_dir / 'report'
    
    csv_file = None
    csv_data = {}
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            csv_path = Path.cwd().parent / 'report' / csv_file
        
        if csv_path.exists():
            print(f"Loading CSV file: {csv_path}")
            csv_data['csv_file'] = str(csv_path)
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('input_filename'):
                        parts = line.split(',')
                        if len(parts) >= 5:
                            input_filename = parts[0].strip()
                            program_name = parts[1].strip()
                            param_group = parts[2].strip()
                            params = parts[3].strip()
                            output_filename = parts[4].strip()
                            key = f"{input_filename}_{program_name}_{param_group}"
                            csv_data[key] = {
                                'input_filename': input_filename,
                                'program_name': program_name,
                                'param_group': param_group,
                                'params': params,
                                'output_filename': output_filename
                            }
            print(f"Loaded {len(csv_data) - 1} records from CSV")
        else:
            print(f"Warning: CSV file not found: {csv_path}")
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    print("Starting image anomaly detection...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if csv_file:
        print(f"CSV reference file: {csv_file}")
    print()
    
    detector = ImageAnomalyDetector(input_dir, output_dir)
    results = detector.evaluate()
    
    print(f"\nEvaluation completed. Total files processed: {len(results)}")
    
    report_path = detector.generate_html_report(report_dir, csv_data)
    print(f"\nReport saved to: {report_path}")
    
    severity_counts = {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 0}
    for result in results:
        severity_counts[result['overall']['severity']] += 1
    
    print("\nSummary:")
    print(f"  Normal: {severity_counts['normal']}")
    print(f"  Mild: {severity_counts['mild']}")
    print(f"  Moderate: {severity_counts['moderate']}")
    print(f"  Severe: {severity_counts['severe']}")


if __name__ == '__main__':
    main()
