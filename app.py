# app.py - 메모리 최적화된 악보 인식 서버
from flask import Flask, request, jsonify
import base64
import io
import json
import gc  # 가비지 컬렉션

app = Flask(__name__)

# 메모리 사용량을 줄이기 위해 필요할 때만 import
def get_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None

def get_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        return None

def get_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
        return Image, ImageDraw, ImageFont
    except ImportError:
        return None, None, None

# 간단한 이미지 처리 함수들
def simple_threshold(image_array, np):
    """간단한 이진화"""
    if len(image_array.shape) == 3:
        # 그레이스케일 변환
        gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = image_array
    
    # 간단한 임계값 처리
    threshold = np.mean(gray)
    binary = (gray < threshold).astype(np.uint8) * 255
    return binary

def find_horizontal_lines(binary_image, np):
    """수평선(오선) 찾기"""
    height, width = binary_image.shape
    lines = []
    
    for row in range(height):
        white_pixels = np.sum(binary_image[row] == 255)
        if white_pixels > width * 0.5:  # 50% 이상이 흰색이면 선으로 간주
            lines.append(row)
    
    return lines

def remove_lines(image, lines, np):
    """선 제거"""
    result = image.copy()
    for line in lines:
        if 0 < line < image.shape[0] - 1:
            # 위아래에 픽셀이 없는 곳만 제거
            for col in range(image.shape[1]):
                if (image[line-1, col] == 0 and image[line+1, col] == 0):
                    result[line, col] = 0
    return result

def find_objects(image, np):
    """간단한 객체 검출"""
    # 연결된 컴포넌트 찾기 (간단한 버전)
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)
    objects = []
    
    def flood_fill(start_row, start_col):
        if (start_row < 0 or start_row >= height or 
            start_col < 0 or start_col >= width or
            visited[start_row, start_col] or 
            image[start_row, start_col] == 0):
            return []
        
        stack = [(start_row, start_col)]
        component = []
        
        while stack:
            row, col = stack.pop()
            if (row < 0 or row >= height or col < 0 or col >= width or
                visited[row, col] or image[row, col] == 0):
                continue
                
            visited[row, col] = True
            component.append((row, col))
            
            # 8방향 연결
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr != 0 or dc != 0:
                        stack.append((row + dr, col + dc))
        
        return component
    
    for row in range(height):
        for col in range(width):
            if image[row, col] == 255 and not visited[row, col]:
                component = flood_fill(row, col)
                if len(component) > 10:  # 최소 크기 필터
                    min_row = min(p[0] for p in component)
                    max_row = max(p[0] for p in component)
                    min_col = min(p[1] for p in component)
                    max_col = max(p[1] for p in component)
                    
                    objects.append({
                        'x': min_col,
                        'y': min_row,
                        'width': max_col - min_col,
                        'height': max_row - min_row,
                        'area': len(component)
                    })
    
    return objects

def analyze_note_positions(objects, staff_lines):
    """음표 위치 분석하여 계이름 결정"""
    notes = []
    
    if len(staff_lines) < 5:
        return notes
    
    # 기본적인 계이름 매핑
    note_names = ['도', '레', '미', '파', '솔', '라', '시']
    
    for obj in objects:
        center_y = obj['y'] + obj['height'] // 2
        
        # 가장 가까운 오선 찾기
        if len(staff_lines) >= 5:
            # 첫 번째 보표의 오선들 사용
            staff_spacing = (staff_lines[4] - staff_lines[0]) / 4 if len(staff_lines) >= 5 else 10
            
            # 두 번째 오선(솔)을 기준으로 계산
            reference_line = staff_lines[1] if len(staff_lines) >= 2 else staff_lines[0]
            distance_from_reference = (reference_line - center_y) / staff_spacing
            
            # 계이름 결정
            note_index = int(round(distance_from_reference)) + 4  # 솔을 기준(4)으로
            
            if 0 <= note_index < len(note_names):
                note_name = note_names[note_index]
            else:
                # 범위를 벗어나면 옥타브 고려
                note_name = note_names[note_index % len(note_names)]
            
            notes.append({
                'x': obj['x'],
                'y': obj['y'],
                'width': obj['width'],
                'height': obj['height'],
                'note_name': note_name,
                'center_x': obj['x'] + obj['width'] // 2,
                'center_y': center_y
            })
    
    return notes

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    cv2 = get_cv2()
    np = get_numpy()
    
    status = {
        'status': 'healthy',
        'opencv': cv2 is not None,
        'numpy': np is not None,
        'message': '악보 인식 서버가 정상 작동 중입니다.'
    }
    
    return jsonify(status)

@app.route('/analyze-sheet-music', methods=['POST'])
def analyze_sheet_music():
    """악보 분석 메인 엔드포인트"""
    cv2 = get_cv2()
    np = get_numpy()
    Image, ImageDraw, ImageFont = get_pil()
    
    if not cv2 or not np:
        return jsonify({
            'success': False,
            'error': 'OpenCV 또는 NumPy가 설치되지 않았습니다.'
        })
    
    try:
        data = request.json
        image_base64 = data['image']
        note_settings = data.get('settings', {})
        
        print("악보 분석 시작...")
        
        # Base64 이미지를 OpenCV 형식으로 변환
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            return jsonify({'success': False, 'error': '이미지를 읽을 수 없습니다.'})
        
        print("이미지 로드 완료")
        
        # 메모리 절약을 위해 이미지 크기 조정
        height, width = original_image.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            original_image = cv2.resize(original_image, (new_width, new_height))
            print(f"이미지 크기 조정: {width}x{height} -> {new_width}x{new_height}")
        
        # 간단한 전처리
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        binary = simple_threshold(gray, np)
        print("이진화 완료")
        
        # 수평선(오선) 검출
        staff_lines = find_horizontal_lines(binary, np)
        print(f"오선 검출 완료: {len(staff_lines)}개")
        
        # 오선 제거
        no_lines_image = remove_lines(binary, staff_lines, np)
        print("오선 제거 완료")
        
        # 객체 검출
        objects = find_objects(no_lines_image, np)
        print(f"객체 검출 완료: {len(objects)}개")
        
        # 음표 분석
        notes = analyze_note_positions(objects, staff_lines)
        print(f"음표 분석 완료: {len(notes)}개")
        
        # 계이름 오버레이
        if Image and ImageDraw:
            result_image = overlay_note_names(original_image, notes, note_settings, Image, ImageDraw, ImageFont)
        else:
            result_image = original_image
        
        print("계이름 오버레이 완료")
        
        # 결과 이미지를 Base64로 변환
        _, buffer = cv2.imencode('.png', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 메모리 정리
        del original_image, gray, binary, no_lines_image
        gc.collect()
        
        return jsonify({
            'success': True,
            'result_image': result_base64,
            'notes_detected': len(notes),
            'notes': notes,
            'staves_detected': len(staff_lines),
            'objects_detected': len(objects)
        })
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        gc.collect()  # 오류 시에도 메모리 정리
        return jsonify({
            'success': False,
            'error': str(e)
        })

def overlay_note_names(original_image, notes, settings, Image, ImageDraw, ImageFont):
    """계이름을 이미지에 오버레이"""
    try:
        # OpenCV 이미지를 PIL로 변환
        pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 폰트 설정
        font_size = settings.get('fontSize', 20)
        font_color = settings.get('fontColor', '#000000')
        
        try:
            # 기본 폰트 사용
            font = ImageFont.load_default()
        except:
            font = None
        
        # 각 음표 아래에 계이름 표시
        for note in notes:
            text = note['note_name']
            x = note['center_x']
            y = note['y'] + note['height'] + 5
            
            if font:
                # 텍스트 크기 계산
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = x - text_width // 2
                
                # 텍스트 그리기
                draw.text((text_x, y), text, fill=font_color, font=font)
            else:
                # 폰트가 없으면 간단한 점으로 표시
                draw.ellipse([x-2, y-2, x+2, y+2], fill=font_color)
        
        # PIL을 다시 OpenCV로 변환
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image
    except:
        return original_image

if __name__ == '__main__':
    print("🎵 악보 인식 서버 시작...")
    print("📍 서버 주소: http://localhost:5000")
    print("🔍 상태 확인: http://localhost:5000/health")
    print("📝 API 엔드포인트: POST /analyze-sheet-music")
    app.run(host='0.0.0.0', port=5000, debug=False)  # debug=False로 메모리 절약
