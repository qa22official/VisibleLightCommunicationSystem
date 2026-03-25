import json
import os
import shutil
import sys
from pathlib import Path

# 在导入 paddleocr 前关闭模型源连通性检查，避免启动时网络探测。
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR

# 加载配置文件
def load_config(config_path='ocr_config.json'):
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"配置文件不存在: {config_file}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_model_dir(path_value):
    model_path = Path(path_value)
    if not model_path.is_absolute():
        model_path = (Path(__file__).resolve().parent / model_path).resolve()

    if not model_path.exists():
        print(f"模型目录不存在: {model_path}")
        sys.exit(1)

    return str(model_path)

# 初始化OCR模型
def init_ocr(config):
    # 验证必需的配置项
    required_keys = [
        'use_doc_orientation_classify',
        'use_doc_unwarping',
        'use_textline_orientation',
        'text_detection_model_dir',
        'text_recognition_model_dir',
        'device'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"配置文件缺失必需项: {', '.join(missing_keys)}")
        sys.exit(1)

    det_model_dir = resolve_model_dir(config['text_detection_model_dir'])
    rec_model_dir = resolve_model_dir(config['text_recognition_model_dir'])

    if bool(config.get('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', False)):
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    
    kwargs = dict(
        use_doc_orientation_classify=config['use_doc_orientation_classify'],
        use_doc_unwarping=config['use_doc_unwarping'],
        use_textline_orientation=config['use_textline_orientation'],
        device=config['device'],
        text_detection_model_dir=det_model_dir,
        text_recognition_model_dir=rec_model_dir,
        text_det_thresh=float(config.get('thresh', 0.3)),
        text_det_box_thresh=float(config.get('box_thresh', 0.5)),
        text_det_unclip_ratio=float(config.get('unclip_ratio', 2.0)),
        char_detection=bool(config.get('char_detection', True)),
    )
    try:
        return PaddleOCR(**kwargs)
    except (TypeError, ValueError) as exc:
        if 'char_detection' not in str(exc):
            raise
        kwargs.pop('char_detection', None)
        return PaddleOCR(**kwargs)


def get_runtime_config(config):
    required_keys = [
        'input_dir',
        'output_dir',
        'image_formats',
        'write_confidence',
        'output_header',
        'separator_length'
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"配置文件缺失必需项: {', '.join(missing_keys)}")
        sys.exit(1)

    if not isinstance(config['image_formats'], list) or not config['image_formats']:
        print("配置项 image_formats 必须是非空数组")
        sys.exit(1)

    return {
        'input_dir': config['input_dir'],
        'output_dir': config['output_dir'],
        'image_formats': {ext.lower() for ext in config['image_formats']},
        'write_confidence': bool(config['write_confidence']),
        'char_detection': bool(config.get('char_detection', True)),
        'output_header': str(config['output_header']),
        'separator_length': int(config['separator_length'])
    }


def extract_text_lines(result):
    text_lines = []
    if not result:
        return text_lines

    # 兼容不同版本返回结构
    for block in result:
        if isinstance(block, dict):
            texts = block.get('rec_texts', [])
            scores = block.get('rec_scores', [])
            for text, score in zip(texts, scores):
                clean_text = str(text).replace(' ', '').replace('\u3000', '')
                text_lines.append(f"{clean_text} (置信度: {float(score):.2f})")
            continue

        if isinstance(block, list):
            for item in block:
                if isinstance(item, list) and len(item) >= 2:
                    rec = item[1]
                    if isinstance(rec, tuple) and len(rec) == 2:
                        text, score = rec
                        clean_text = str(text).replace(' ', '').replace('\u3000', '')
                        text_lines.append(f"{clean_text} (置信度: {float(score):.2f})")

    return text_lines

# 处理图片文件
def process_images(ocr, runtime_config):
    input_path = Path(runtime_config['input_dir'])
    output_path = Path(runtime_config['output_dir'])

    if not input_path.exists():
        print(f"输入目录不存在: {input_path}")
        return
    
    # 创建输出目录（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_formats = runtime_config['image_formats']
    
    # 扫描input文件夹中的所有图片
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_formats]
    
    if not image_files:
        print(f"在 '{runtime_config['input_dir']}' 文件夹中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始处理...\n")
    
    # 处理每张图片
    for image_file in sorted(image_files):
        try:
            print(f"处理: {image_file.name}")
            
            # 执行OCR
            result = ocr.predict(str(image_file), return_word_box=runtime_config['char_detection'])
            
            # 提取文本
            text_lines = extract_text_lines(result)
            
            # 保存为txt文件
            output_file = output_path / f"{image_file.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"{runtime_config['output_header']}{image_file.name}\n")
                f.write("=" * runtime_config['separator_length'] + "\n\n")
                
                if text_lines:
                    if runtime_config['write_confidence']:
                        for line in text_lines:
                            f.write(line + "\n")
                    else:
                        for line in text_lines:
                            f.write(line.split(' (置信度:', 1)[0] + "\n")
                else:
                    f.write("未识别到文本\n")
            
            print(f"已保存到: {output_file.name}\n")
        
        except Exception as e:
            print(f"处理失败: {image_file.name} - {str(e)}\n")


def cleanup_pycache(root: Path) -> None:
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

# 主程序
if __name__ == '__main__':
    print("=" * 50)
    print("PaddleOCR 批量处理系统")
    print("=" * 50 + "\n")

    try:
        try:
            # 加载配置
            print("加载配置文件...")
            config = load_config('ocr_config.json')
            print("配置加载成功\n")
            runtime_config = get_runtime_config(config)

            # 初始化OCR
            print("初始化OCR模型...")
            ocr = init_ocr(config)
            print("模型初始化成功\n")

            # 处理图片
            process_images(ocr, runtime_config)

            print("=" * 50)
            print("所有处理完成！")
            print("=" * 50)

        except Exception as e:
            print(f"错误: {str(e)}")
    finally:
        cleanup_pycache(Path(__file__).resolve().parent)