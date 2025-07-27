# app.py - 악보 인식 서버
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import json

app = Flask(__name__)

# 기본 함수들 (앞서 제공해주신 코드 기반)
def threshold(image):
    """이미지 이진화"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image

def weighted(value, standard=10):
    """가중치 계산"""
    return int(value * (standard / 10))

def closing(image, standard=10):
    """닫힘 연산"""
    kernel = np.ones((weighted(5, standard), weighted(5, standard)), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def get_center(y, h):
    """객체 중심점 계산"""
    return (y + y + h) / 2

def remove_noise(image):
    """보표 영역만 추출"""
    image = threshold(image)
    mask = np.zeros(image.shape, np.uint8)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    
    for i in range(1, cnt):
        x, y, w, h, area = stats[i]
        if w > image.shape[1] * 0.5:  # 이미지 넓이의 50% 이상
            cv2.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)
    
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def remove_staves(image):
    """오선 제거"""
    height, width = image.shape
    staves = []
    
    # 오선 검출
    for row in range(height):
        pixels = 0
        for col in range(width):
            pixels += (image[row][col] == 255)
        
        if pixels >= width * 0.5:
            if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:
                staves.append([row, 1])
            else:
                staves[-1][1] += 1
    
    # 오선 제거
    for staff in range(len(staves)):
        top_pixel = staves[staff][0]
        bot_pixel = staves[staff][0] + staves[staff][1]
        
        for col in range(width):
            if (top_pixel > 0 and bot_pixel < height-1 and 
                image[top_pixel - 1][col] == 0 and 
                image[bot_pixel + 1][col] == 0):
                for row in range(top_pixel, bot_pixel + 1):
                    image[row][col] = 0
    
    return image, [x[0] for x in staves]

def normalization(image, staves, standard=10):
    """이미지 정규화"""
    if len(staves) < 5:
        return image, staves
    
    avg_distance = 0
    lines = int(len(staves) / 5)
    
    for line in range(lines):
        for staff in range(4):
            if line * 5 + staff + 1 < len(staves):
                staff_above = staves[line * 5 + staff]
                staff_below = staves[line * 5 + staff + 1]
                avg_distance += abs(staff_above - staff_below)
    
    if avg_distance == 0:
        return image, staves
    
    avg_distance /= len(staves) - lines
    height, width = image.shape
    weight = standard / avg_distance
    
    new_width = int(width * weight)
    new_height = int(height * weight)
    
    image = cv2.resize(image, (new_width, new_height))
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    staves = [x * weight for x in staves]
    
    return image, staves

def object_detection(image, staves, standard=10):
    """객체 검출"""
    if len(staves) < 5:
        return image, []
    
    lines = int(len(staves) / 5)
    objects = []
    closing_image = closing(image, standard)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image)
    
    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i]
        if w >= weighted(5, standard) and h >= weighted(5, standard):
            center = get_center(y, h)
            for line in range(lines):
                area_top = staves[line * 5] - weighted(20, standard)
                area_bot = staves[(line + 1) * 5 - 1] + weighted(20, standard)
                if area_top <= center <= area_bot:
                    objects.append([line, (x, y, w, h, area)])
    
    objects.sort()
    return image, objects

def analyze_note_positions(objects, staves, standard=10):
    """음표 위치 분석하여 계이름 결정"""
    notes = []
    
    if len(staves) < 5:
        return notes
    
    # 기본적인 계이름 매핑 (트레블 클레프 기준)
    # 오선 간격을 기준으로 음높이 계산
    for obj in objects:
        line_num, (x, y, w, h, area) = obj
        
        if line_num * 5 + 4 >= len(staves):
            continue
            
        # 해당 보표의 오선들
        staff_lines = staves[line_num * 5:line_num * 5 + 5]
        center_y = y + h // 2
        
        # 오선 간격 계산
        if len(staff_lines) >= 2:
            line_spacing = (staff_lines[-1] - staff_lines[0]) / 4
            
            # 기준선(두 번째 오선, 트레블 클레프의 솔)에서의 거리
            reference_line = staff_lines[1]  # 두 번째 오선 (솔)
            distance_from_sol = (reference_line - center_y) / line_spacing
            
            # 거리에 따른 계이름 결정
            note_names = ['도', '레', '미', '파', '솔', '라', '시']
            
            # 솔(두 번째 오선)을 기준(인덱스 4)으로 계산
            note_index = int(round(distance_from_sol)) + 4
            
            # 범위 내에서 계이름 결정
            if 0 <= note_index < len(note_names):
                note_name = note_names[note_index]
            else:
                # 범위를 벗어나면 옥타브 계산
                octave_offset = note_index // len(note_names)
                adjusted_index = note_index % len(note_names)
                note_name = note_names[adjusted_index]
                if octave_offset > 0:
                    note_name += f"(+{octave_offset})"
                elif octave_offset < 0:
                    note_name += f"({octave_offset})"
            
            notes.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'note_name': note_name,
                'center_x': x + w // 2,
                'center_y': center_y
            })
    
    return notes

def overlay_note_names(original_image, notes, settings):
    """계이름을 이미지에 오버레이"""
    # OpenCV 이미지를 PIL로 변환
    pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 폰트 설정
    font_size = settings.get('fontSize', 24)
    font_color = settings.get('fontColor', '#000000')
    
    try:
        # 시스템 기본 폰트 사용
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 각 음표 아래에 계이름 표시
    for note in notes:
        text = note['note_name']
        x = note['center_x']
        y = note['y'] + note['height'] + 5  # 음표 아래쪽에 표시
        
        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x - text_width // 2  # 중앙 정렬
        
        # 텍스트 그리기
        draw.text((text_x, y), text, fill=font_color, font=font)
    
    # PIL을 다시 OpenCV로 변환
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result_image

@app.route('/analyze-sheet-music', methods=['POST'])
def analyze_sheet_music():
    """악보 분석 메인 엔드포인트"""
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
        
        # 악보 인식 처리 과정
        # 1. 전처리
        processed_image = remove_noise(original_image.copy())
        print("노이즈 제거 완료")
        
        # 2. 오선 제거
        no_staves_image, staves = remove_staves(processed_image.copy())
        print(f"오선 제거 완료, 검출된 오선: {len(staves)}개")
        
        # 3. 정규화
        normalized_image, normalized_staves = normalization(no_staves_image.copy(), staves.copy())
        print("정규화 완료")
        
        # 4. 객체 검출
        final_image, objects = object_detection(normalized_image.copy(), normalized_staves.copy())
        print(f"객체 검출 완료, 검출된 객체: {len(objects)}개")
        
        # 5. 음표 위치 분석
        notes = analyze_note_positions(objects, normalized_staves)
        print(f"음표 분석 완료, 분석된 음표: {len(notes)}개")
        
        # 6. 계이름 오버레이
        result_image = overlay_note_names(original_image, notes, note_settings)
        print("계이름 오버레이 완료")
        
        # 결과 이미지를 Base64로 변환
        _, buffer = cv2.imencode('.png', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'result_image': result_base64,
            'notes_detected': len(notes),
            'notes': notes,
            'staves_detected': len(staves),
            'objects_detected': len(objects)
        })
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({'status': 'healthy', 'message': '악보 인식 서버가 정상 작동 중입니다.'})

if __name__ == '__main__':
    print("🎵 악보 인식 서버 시작...")
    print("📍 서버 주소: http://localhost:5000")
    print("🔍 상태 확인: http://localhost:5000/health")
    print("📝 API 엔드포인트: POST /analyze-sheet-music")
    app.run(host='0.0.0.0', port=5001, debug=True)