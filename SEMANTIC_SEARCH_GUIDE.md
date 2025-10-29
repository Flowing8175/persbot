# Semantic Search 사용 가이드

## 📌 개요

Semantic Search(의미론적 검색)는 사용자의 기억 중에서 **현재 대화와 의미적으로 관련된 기억들만 골라서** 로드하는 고급 기능입니다.

### 기본 모드 vs 시맨틱 서치

```
┌─────────────────────────────────────────────────────────┐
│              Inject All (기본 모드)                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ 📋 가장 최근의 20개 기억을 모두 불러옴                  │
│                                                          │
│ 장점:                                                   │
│ ✅ 빠름 (100ms 이하)                                   │
│ ✅ 리소스 적음                                          │
│ ✅ 1GB RAM 환경에 최적화                                │
│                                                          │
│ 단점:                                                   │
│ ❌ 무관한 기억도 로드됨                                 │
│ ❌ 토큰 낭비                                           │
│                                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│           Semantic Search (시맨틱 서치)                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ 🎯 현재 대화와 관련된 기억만 지능적으로 검색            │
│                                                          │
│ 장점:                                                   │
│ ✅ 정확한 기억 로드                                     │
│ ✅ 관련 없는 기억은 제외                                │
│ ✅ 토큰 효율적                                         │
│ ✅ 더 맥락있는 응답                                    │
│                                                          │
│ 단점:                                                   │
│ ❌ 느림 (500-1000ms)                                  │
│ ❌ 리소스 사용 (모델 로드 필요)                         │
│ ❌ 첫 사용 시 모델 다운로드 필요                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Semantic Search 활성화하기

### 방법 1: 명령어로 활성화 (추천)

Discord에서 다음 명령어를 입력하세요:

```
!기억설정 semantic_search
```

**응답:**
```
✓ 기억 검색 모드를 semantic_search로 변경했습니다.
```

---

### 방법 2: 설정 파일에서 활성화

`.env` 파일을 수정하는 대신, 코드에서 직접 설정할 수 있습니다.

**`config.py` 수정:**
```python
@dataclass(frozen=True)
class AppConfig:
    # ... 기존 설정 ...
    memory_retrieval_mode: str = 'semantic_search'  # 'inject_all'에서 변경
```

봇을 재시작하면 적용됩니다.

---

### 방법 3: 프로그래밍으로 설정

Python 코드에서 실시간으로 변경:

```python
# memory_service 객체 존재할 때
memory_service.set_retrieval_mode('semantic_search')
```

---

## 📊 작동 원리

### Step 1: 사용자 메시지 수신

```
User: "@소예봇 게임 개발에서 버그를 어떻게 처리해?"
```

### Step 2: 임베딩 모델 로드

```
현재 모드가 semantic_search이므로:
1. 임베딩 모델 로드 (all-MiniLM-L6-v2)
   - 파일 크기: ~80MB
   - 로드 시간: 2-3초 (처음 1회만)
   - 이후: 메모리에 캐시됨

2. 사용자 메시지 → 벡터로 변환
   "게임 개발에서 버그를 어떻게 처리해?"
   ↓
   [0.234, -0.567, 0.890, ..., 0.123]  (384차원 벡터)
```

### Step 3: 사용자의 모든 기억 로드

```
데이터베이스에서 사용자의 모든 기억 조회:

Memory 1: "사용자는 게임 개발자다" (type: fact)
Memory 2: "사용자는 C++ 좋아한다" (type: preference)
Memory 3: "사용자는 빠른 응답 원한다" (type: preference)
Memory 4: "사용자 생일: 12월 25일" (type: key_memory)
Memory 5: "좋아하는 게임: 발로란트" (type: fact)
... (더 많을 수 있음)
```

### Step 4: 각 기억을 벡터로 변환

```
각 기억마다:
1. 텍스트 → 벡터 변환
2. 임베딩 저장 (또는 캐시에서 로드)

Memory 1 벡터: [0.123, 0.456, -0.789, ..., 0.234]
Memory 2 벡터: [-0.456, 0.123, 0.567, ..., 0.890]
Memory 3 벡터: [0.789, -0.234, 0.123, ..., -0.567]
... (모든 기억)
```

### Step 5: 유사도 계산 (Cosine Similarity)

```
사용자 메시지 벡터와 각 기억 벡터의 유사도 계산:

Memory 1: "게임 개발자" vs "게임 개발 질문"
         유사도 = 0.89  ⭐⭐⭐ (매우 높음!)

Memory 2: "C++ 좋아한다" vs "게임 개발 질문"
         유사도 = 0.72  ⭐⭐ (중간)

Memory 3: "빠른 응답 원한다" vs "게임 개발 질문"
         유사도 = 0.45  ⭐ (낮음)

Memory 4: "생일: 12월 25일" vs "게임 개발 질문"
         유사도 = 0.12  (매우 낮음)

Memory 5: "좋아하는 게임: 발로란트" vs "게임 개발 질문"
         유사도 = 0.78  ⭐⭐ (높음)
```

### Step 6: 상위 K개 선택

```
유사도순으로 정렬:
1. Memory 1: 0.89 ← 선택 ✅
2. Memory 5: 0.78 ← 선택 ✅
3. Memory 2: 0.72 ← 선택 ✅
4. Memory 3: 0.45 (선택 안 함 ❌)
5. Memory 4: 0.12 (선택 안 함 ❌)

결과: 상위 3개 기억만 Gemini에 전달
(설정에서 max_memories_to_inject = 20이면 최대 20개)
```

### Step 7: 관련 기억들을 프롬프트에 주입

```
Gemini에게 전달:

[시스템 프롬프트]
당신은 소예입니다...

[메모리 컨텍스트] ← 검색된 것만!
🧠 당신이 알고 있는 정보:

### Facts
- 사용자는 게임 개발자다

### Preferences
- 사용자는 빠른 응답을 원한다
- 사용자는 C++을 좋아한다

### Key Memories
- 사용자가 좋아하는 게임: 발로란트

[사용자 메시지]
게임 개발에서 버그를 어떻게 처리해?
```

### Step 8: Gemini 응답

```
Gemini의 생각:
"아, 이 사용자는 게임 개발자고, C++을 좋아하고,
게임 개발 관련 질문이네. 게임 개발 지식으로 답변해야겠다"

응답:
"게임 개발에서는 보통 디버거나 로그로 버그를 추적해.
C++이면 Visual Studio의 디버거가 좋아.
발로란트 같은 실제 게임들도 이렇게 처리하거든."
```

### Step 9: 임베딩 모델 언로드

```
더 이상 필요 없으므로:
1. 임베딩 모델 메모리에서 제거
2. 약 150MB RAM 해제
3. 다음 요청 시 다시 로드
```

---

## ⚙️ 설정 및 커스터마이징

### 기본 설정

```python
# config.py의 기본값
memory_retrieval_mode: str = 'inject_all'        # 기본: 간단한 모드
memory_retrieval_mode: str = 'semantic_search'  # 변경: 시맨틱 서치

max_memories_to_inject: int = 20                # 한 번에 로드할 최대 기억 수
embedding_model_name: str = 'all-MiniLM-L6-v2' # 사용할 임베딩 모델
```

### 커스터마이징

**더 많은 기억 로드하기:**
```python
# config.py에서
max_memories_to_inject: int = 50  # 20 → 50으로 증가
```

**다른 임베딩 모델 사용:**
```python
# 매우 가벼운 모델
embedding_model_name: str = 'all-MiniLM-L6-v2'  # 기본 (80MB)

# 더 정확한 모델
embedding_model_name: str = 'all-mpnet-base-v2'  # 더 크지만 정확 (430MB)
```

---

## 🎯 사용 시나리오

### 시나리오 1: 사용자가 매우 많은 기억을 가진 경우

```
상황:
- 사용자가 100개 이상의 기억을 저장함
- Inject All 모드: 매번 20개 모두 로드 → 토큰 낭비
- Semantic Search 모드: 관련된 5-10개만 로드 → 효율적

결과:
- Gemini API 비용 감소
- 응답 속도 향상
- 메모리 사용량 감소
```

### 시나리오 2: 여러 주제의 기억이 섞여 있는 경우

```
사용자의 기억:
1. "프로그래밍을 좋아한다"
2. "요리에 관심이 있다"
3. "게임 개발자다"
4. "커피를 좋아한다"
5. "독서광이다"
6. "여행을 좋아한다"
... (100개)

User: "게임 개발할 때 최적화는 어떻게 하지?"

Inject All: 모든 100개 로드 (비효율)
           → AI가 "커피" "요리" "여행" 등도 고려

Semantic Search: 관련 것만 로드 (효율적)
                → AI가 "프로그래밍" "게임 개발" 등만 참고
```

### 시나리오 3: 실시간 토픽 변경

```
User: "@소예봇 C++ 포인터 설명해줄 수 있어?"
→ C++ 관련 기억만 로드

User: "@소예봇 너 요즘 뭐 하고 있었어?"
→ 최근 상호작용 기억 로드

User: "@소예봇 내 생일 기억해?"
→ 생일 관련 기억만 로드

(모든 상황에 최적화된 기억 로드!)
```

---

## 📈 성능 비교

### 속도 비교

```
작업                    Inject All      Semantic Search
─────────────────────────────────────────────────────────
기억 로드              5-10ms          5-10ms
임베딩 모델 로드       0ms             2-3초 (첫 1회)
                      (캐시됨)
벡터 변환             0ms             100-300ms
유사도 계산           0ms             50-100ms
프롬프트 생성         10-20ms         10-20ms
─────────────────────────────────────────────────────────
총 시간               20-30ms         2.2-3.5초 (첫 1회)
                                      200-500ms (이후)
```

### 리소스 사용 비교

```
항목                   Inject All      Semantic Search
─────────────────────────────────────────────────────────
기본 RAM 사용         0MB 추가        0MB 추가
임베딩 모델 로드 시   0MB            150MB (임시)
데이터베이스 쿼리     간단함         복잡함
네트워크 대역폭       적음           적음
Gemini API 토큰      많음           적음
─────────────────────────────────────────────────────────
권장 환경            1GB RAM         2GB RAM 이상
```

---

## 🔧 문제 해결

### 문제 1: "Semantic Search 모드로 바꾸면 에러가 난다"

**원인:**
- sentence-transformers 라이브러리가 설치되지 않음

**해결:**
```bash
pip install sentence-transformers
```

---

### 문제 2: "Semantic Search 매우 느려진다"

**원인:**
- 첫 사용 시 모델 다운로드 (200-500MB)
- CPU가 약함

**해결:**
```python
# 더 가벼운 모델 사용
# config.py에서
embedding_model_name: str = 'distiluse-base-multilingual-cased-v2'  # 더 빠름
```

또는 Inject All 모드로 돌아가기:
```
!기억설정 inject_all
```

---

### 문제 3: "메모리 부족으로 모델 로드 실패"

**증상:**
```
MemoryError: Unable to allocate X GiB for an array...
```

**해결:**
```
!기억설정 inject_all
```

(Inject All 모드는 모델을 로드하지 않으므로 메모리 안전)

---

### 문제 4: "임베딩 모델이 계속 로드/언로드되어서 느려진다"

**원인:**
- 매 요청마다 모델을 로드/언로드 중

**확인:**
```
로그에서:
"Loading embedding model: all-MiniLM-L6-v2"
"Embedding model loaded successfully"
...
"Embedding model unloaded"
(이 패턴이 반복됨)
```

**해결:**
- 이것은 정상입니다 (메모리 절약 설계)
- 더 빠르게 하려면 모델을 메모리에 계속 로드:

```python
# services/memory_service.py 수정
def __init__(self, ...):
    # ...
    self._load_embedding_model()  # 항상 로드된 상태 유지
    # (한 번만 로드됨)
```

---

## 💡 Best Practices

### 1. 적절한 모드 선택

```
상황별 추천:

📱 1GB RAM / 1-Core vCPU
   → Inject All 모드 (권장)

💻 2GB RAM / 2-Core vCPU
   → Inject All 모드 (안전)

🖥️ 4GB RAM 이상 / 4-Core 이상
   → Semantic Search 모드 (권장)
```

### 2. 모드 전환 시점

```
언제 Semantic Search로 전환할까?

✅ 바꿔도 좋은 상황:
   - 사용자가 50개 이상의 기억 보유
   - 다양한 주제의 기억들이 섞여 있음
   - API 비용 절감이 중요함
   - 응답 정확도를 높이고 싶음

❌ 바꾸면 안 되는 상황:
   - RAM이 1GB 이하
   - 실시간성이 매우 중요함
   - 첫 응답이 느리면 안 됨
   - CPU가 매우 약함
```

### 3. 기억 관리

Semantic Search 효율을 높이려면:

```
좋은 기억 저장:
✅ "사용자는 C++ 개발자다"
✅ "사용자는 게임 개발을 좋아한다"
✅ "사용자는 프로젝트 마감이 내일이다"

나쁜 기억 저장:
❌ "오늘 날씨 좋네"
❌ "뭐 먹지"
❌ "암호"

이유:
- 좋은 기억: 미래 대화에 정말 유용
- 나쁜 기억: 토큰 낭비, 검색 효율 저하
```

### 4. 정기적인 기억 정리

```
매달 1회:
!기억목록
로 현재 기억들 확인

불필요한 기억 삭제:
!기억삭제 [ID]

예: 3개월 전에 "내일 마감"이라는 기억 → 이미 지났으므로 삭제
```

---

## 📚 고급 활용

### 커스텀 임베딩 모델 추가

더 정확하거나 빠른 모델을 사용하고 싶다면:

```python
# 사용 가능한 모델들:

# 매우 가벼움 (빠름)
'all-MiniLM-L6-v2'           # 22M 파라미터, 80MB

# 일반적 (균형)
'paraphrase-MiniLM-L6-v2'   # 22M 파라미터, 80MB

# 정확함 (느림)
'all-mpnet-base-v2'         # 109M 파라미터, 430MB

# 한국어 최적화
'sentence-transformers/xlm-r-base-multilingual'  # 250MB
```

**변경 방법:**
```python
# config.py에서
embedding_model_name: str = 'paraphrase-MiniLM-L6-v2'
```

---

### 임베딩 캐싱

이미 계산한 임베딩은 저장되어 있습니다:

```python
# database/models.py의 Memory 모델
embedding = Column(LargeBinary, nullable=True)
```

자동으로 계산되고 저장되므로 다음 번엔 더 빠릅니다.

---

## 📊 모니터링

### Semantic Search 상태 확인

```python
# memory_service 상태 확인
current_mode = memory_service.retrieval_mode
# 출력: 'semantic_search' 또는 'inject_all'

# 임베딩 모델 로드 여부
if memory_service.embedding_model is not None:
    print("모델 로드됨 ✅")
else:
    print("모델 언로드됨 (필요시 로드) ⏳")
```

### 성능 로깅

로그에서 Semantic Search 작동 확인:

```
[INFO] Loading embedding model: all-MiniLM-L6-v2
[INFO] Embedding model loaded successfully
[DEBUG] Semantic search: 5 relevant memories found
[DEBUG] Embedding model unloaded
```

---

## 🎓 학습 자료

### 임베딩이란?

```
텍스트 → 벡터 변환

예시:
"게임 개발자"   → [0.23, -0.45, 0.67, ..., 0.12]  (384차원)
"프로그래밍"    → [0.25, -0.43, 0.65, ..., 0.14]  (384차원)
"요리사"        → [-0.10, 0.80, -0.30, ..., 0.92] (384차원)

특징:
- 비슷한 의미 → 벡터도 비슷함
- "게임 개발자" vs "프로그래밍" = 유사도 높음
- "게임 개발자" vs "요리사" = 유사도 낮음
```

### Cosine Similarity란?

```
두 벡터의 각도를 이용한 유사도 측정

유사도 = 1.0   → 완전히 같음
유사도 = 0.5   → 중간 정도로 관련
유사도 = 0.0   → 완전히 다름
유사도 = -1.0  → 정반대

예시:
"게임 개발" vs "게임 개발" = 1.0 ✅ 완전 일치
"게임 개발" vs "프로그래밍" = 0.82 ⭐ 매우 관련
"게임 개발" vs "요리" = 0.15 ❌ 거의 무관
```

---

## 🚀 추천 설정

### 환경별 최적 설정

**환경 1: 1GB RAM / 1-Core vCPU (Oracle Cloud Always Free)**
```python
memory_retrieval_mode = 'inject_all'        # 중요!
max_memories_to_inject = 15                 # 작게 유지
enable_memory_system = True                 # 기억은 활성화
```

**환경 2: 2GB RAM / 2-Core vCPU**
```python
memory_retrieval_mode = 'inject_all'        # 안전성 우선
max_memories_to_inject = 20
enable_memory_system = True
```

**환경 3: 4GB RAM 이상 / 4-Core 이상**
```python
memory_retrieval_mode = 'semantic_search'   # 시맨틱 활성화!
max_memories_to_inject = 30
embedding_model_name = 'all-MiniLM-L6-v2'
enable_memory_system = True
```

---

## 📞 요약

### 한 눈에 보기

| 기능 | Inject All | Semantic Search |
|------|-----------|-----------------|
| 속도 | ⚡⚡⚡ (빠름) | ⚡ (느림) |
| 정확도 | ⭐⭐ (보통) | ⭐⭐⭐ (높음) |
| RAM | 🟢 (적음) | 🟡 (중간) |
| 복잡도 | 🟢 (간단) | 🔴 (복잡) |
| 추천 환경 | 1GB RAM | 4GB+ RAM |

### 빠른 시작

1. **현재 모드 확인:**
   ```
   !기억설정
   ```

2. **Semantic Search 시도:**
   ```
   !기억설정 semantic_search
   ```

3. **문제 발생하면 복구:**
   ```
   !기억설정 inject_all
   ```

4. **상태 확인:**
   ```
   !기억통계
   ```

---

## 🎯 결론

- **Inject All**: 안정적, 빠르고 간단 → **대부분 사용자에게 권장**
- **Semantic Search**: 정확함, 리소스 사용 → **고사양 환경에서 추천**

두 모드 모두 언제든 전환 가능하므로, 자신의 환경에 맞게 선택하세요!
