import re
import collections
import argparse
import os
from konlpy.tag import Okt  # 한국어 형태소 분석기
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

# Usage:
# Library : pip install konlpy matplotlib wordcloud pandas 
# Command(원문) : python word_frequency_analyzer.py ko_novel.txt --raw_text --min_word_length 5 --max_word_length 15
# Command(단어) : python word_frequency_analyzer.py --min_word_length 2 --ngram_size 6 ko_novel.txt

def clean_text(text):
    """텍스트 전처리 함수"""
    # 특수 문자 및 숫자 제거
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # 여러 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_words(text, min_length=2):
    """텍스트에서 단어 추출"""
    okt = Okt()
    words = []
    
    # 형태소 분석
    morphs = okt.pos(text)
    
    # 명사, 동사, 형용사만 추출 (원하는 품사를 선택할 수 있음)
    for word, pos in morphs:
        if pos.startswith('N') or pos.startswith('V') or pos.startswith('XR'):  # 명사, 동사, 형용사
            if len(word) >= min_length:  # 최소 길이 이상인 단어만 포함
                words.append(word)
    
    return words

def extract_raw_text_units(text, min_length=2, max_length=15):
    """텍스트에서 원문 단위(띄어쓰기 단위) 추출 - 글자수 기준으로 필터링"""
    # 텍스트를 띄어쓰기 단위로 분리
    text_units = text.split()
    
    # 지정된 글자수 범위 내의 단위만 포함
    return [unit for unit in text_units if min_length <= len(unit) <= max_length]

def extract_phrases(text, n=2, min_count=5):
    """n-gram 표현 추출"""
    words = extract_words(text)
    phrases = []
    
    for i in range(len(words) - n + 1):
        phrase = ' '.join(words[i:i+n])
        phrases.append(phrase)
    
    # 최소 빈도수 이상인 표현만 반환
    counter = collections.Counter(phrases)
    return {phrase: count for phrase, count in counter.items() if count >= min_count}

def create_output_folder(file_path, output_dir=None):
    """결과 저장을 위한 폴더 생성"""
    if output_dir is None:
        # 파일 이름에서 확장자 제거
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        
        # 결과 폴더 경로 생성 (파일명_results)
        output_dir = f"{file_name_without_ext}_results"
    
    # 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"결과 저장 폴더 생성: {output_dir}")
    
    return output_dir

def analyze_file(file_path, top_n=50, min_word_length=2, max_word_length=15, ngram_size=2, min_phrase_count=5, analyze_raw_text=False, output_dir=None):
    """파일 분석 및 결과 출력"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # UTF-8로 읽기 실패시 다른 인코딩 시도
        with open(file_path, 'r', encoding='cp949') as file:
            text = file.read()
    
    # 결과 저장 폴더 생성
    output_dir = create_output_folder(file_path, output_dir)
    
    # 파일 이름 가져오기 (경로 제외)
    base_name = os.path.basename(file_path)
    
    # 원문 분석 모드 (형태소 분석 없이 글자수 기준)
    if analyze_raw_text:
        # 텍스트 전처리 (특수문자 제거는 유지)
        # 여러 공백을 하나로 치환
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # 원문 단위 추출 및 빈도 계산
        text_units = extract_raw_text_units(cleaned_text, min_length=min_word_length, max_length=max_word_length)
        unit_counts = collections.Counter(text_units)
        
        # 결과 출력
        print(f"\n가장 많이 나오는 원문 단위 (글자수 {min_word_length}~{max_word_length}자, 상위 {top_n}개):")
        for unit, count in unit_counts.most_common(top_n):
            print(f"{unit}: {count}회")
        
        # 결과를 데이터프레임으로 변환하여 CSV로 저장
        unit_df = pd.DataFrame(unit_counts.most_common(top_n), columns=['원문 단위', '빈도수'])
        csv_path = os.path.join(output_dir, f"{base_name}_raw_text_frequency.csv")
        unit_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"원문 단위 빈도 CSV 파일 저장: {csv_path}")
        
        # 워드클라우드 생성
        try:
            wordcloud = WordCloud(
                font_path='NanumGothic.ttf',  # 한글 폰트 경로 (필요시 수정)
                width=800, 
                height=400, 
                background_color='white'
            ).generate_from_frequencies(dict(unit_counts.most_common(100)))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            
            wordcloud_path = os.path.join(output_dir, f"{base_name}_raw_text_wordcloud.png")
            plt.savefig(wordcloud_path, dpi=300)
            print(f"원문 단위 워드클라우드 이미지 저장: {wordcloud_path}")
        except Exception as e:
            print(f"워드클라우드 생성 중 오류 발생: {e}")
        
        return unit_counts.most_common(top_n)
    
    # 기존 단어 분석 모드
    else:
        # 텍스트 전처리
        cleaned_text = clean_text(text)
        
        # 단어 추출 및 빈도 계산
        words = extract_words(cleaned_text, min_length=min_word_length)
        word_counts = collections.Counter(words)
        
        # 표현(n-gram) 추출 및 빈도 계산
        phrases = extract_phrases(cleaned_text, n=ngram_size, min_count=min_phrase_count)
        
        # 결과 출력
        print(f"\n가장 많이 나오는 단어 (상위 {top_n}개):")
        for word, count in word_counts.most_common(top_n):
            print(f"{word}: {count}회")
        
        print(f"\n가장 많이 나오는 {ngram_size}-gram 표현 (최소 {min_phrase_count}회 이상):")
        for phrase, count in sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            print(f"{phrase}: {count}회")
        
        # 결과를 데이터프레임으로 변환하여 CSV로 저장
        word_df = pd.DataFrame(word_counts.most_common(top_n), columns=['단어', '빈도수'])
        csv_path = os.path.join(output_dir, f"{base_name}_word_frequency.csv")
        word_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"단어 빈도 CSV 파일 저장: {csv_path}")
        
        # n-gram 결과 저장
        phrase_df = pd.DataFrame(
            [(phrase, count) for phrase, count in sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:top_n]],
            columns=[f'{ngram_size}-gram 표현', '빈도수']
        )
        ngram_csv_path = os.path.join(output_dir, f"{base_name}_ngram{ngram_size}_frequency.csv")
        phrase_df.to_csv(ngram_csv_path, index=False, encoding='utf-8-sig')
        print(f"n-gram 빈도 CSV 파일 저장: {ngram_csv_path}")
        
        # 워드클라우드 생성
        try:
            wordcloud = WordCloud(
                font_path='NanumGothic.ttf',  # 한글 폰트 경로 (필요시 수정)
                width=800, 
                height=400, 
                background_color='white'
            ).generate_from_frequencies(dict(word_counts.most_common(100)))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            
            wordcloud_path = os.path.join(output_dir, f"{base_name}_wordcloud.png")
            plt.savefig(wordcloud_path, dpi=300)
            print(f"단어 워드클라우드 이미지 저장: {wordcloud_path}")
        except Exception as e:
            print(f"워드클라우드 생성 중 오류 발생: {e}")
        
        return word_counts.most_common(top_n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='텍스트 파일에서 가장 많이 나오는 단어와 표현을 분석합니다.')
    parser.add_argument('file_path', help='분석할 텍스트 파일 경로')
    parser.add_argument('--top_n', type=int, default=50, help='출력할 상위 단어/표현 개수 (기본값: 50)')
    parser.add_argument('--min_word_length', type=int, default=2, help='분석할 최소 단어/텍스트 길이 (기본값: 2)')
    parser.add_argument('--max_word_length', type=int, default=15, help='분석할 최대 텍스트 길이 (기본값: 15, 원문 분석 모드에서만 적용)')
    parser.add_argument('--ngram_size', type=int, default=2, help='n-gram 크기 (기본값: 2)')
    parser.add_argument('--min_phrase_count', type=int, default=5, help='표현의 최소 출현 횟수 (기본값: 5)')
    parser.add_argument('--raw_text', action='store_true', help='원문 분석 모드 활성화 - 형태소 분석 없이 지정된 글자수 범위의 단위를 분석')
    parser.add_argument('--output_dir', help='결과를 저장할 폴더 경로 (지정하지 않으면 자동 생성)')
    
    args = parser.parse_args()
    
    print(f"'{args.file_path}' 파일 분석 중...")
    analyze_file(
        args.file_path, 
        args.top_n, 
        args.min_word_length,
        args.max_word_length,
        args.ngram_size, 
        args.min_phrase_count,
        args.raw_text,  # eojeols 대신 raw_text로 변경
        args.output_dir
    )
    print(f"\n분석 완료! 결과가 저장되었습니다.") 