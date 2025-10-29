# SoyeBot 코드베이스 정리 - 메시지 처리 흐름

## 📊 시스템 개요

SoyeBot은 Discord 봇으로, 사용자가 메시지를 보낼 때 다음과 같은 과정을 거쳐 응답을 생성합니다:

```
Discord 메시지 → 봇 수신 → 처리 → AI 응답 생성 → Discord에 전송
                                   ↓
                              기억 저장 (선택)
```

---

## 🔄 메시지 처리의 논리적 흐름

### 1단계: 메시지 도착 및 수신

**어디서:** `bot/cogs/assistant.py` - `on_message()` 함수

**무엇을 하는가:**
```python
@commands.Cog.listener()
async def on_message(self, message: discord.Message):
    # 1. 봇이 멘션되었는지 확인
    if message.author.bot or not self.bot.user.mentioned_in(message):
        return  # 무시

    # 2. 메시지 텍스트 추출
    user_message = extract_message_content(message)
    if not user_message:
        return  # 메시지가 비어있으면 무시
```

**예시:**
- 사용자: `@소예봇 안녕하세요!`
- 봇: "아, @mention이 있네. 처리해야겠다"

---

### 2단계: 세션 생성 또는 기존 세션 재사용

**어디서:** `bot/session.py` - `SessionManager.get_or_create()`

**무엇을 하는가:**
```python
# 사용자별로 하나의 대화 세션을 관리
chat_session, user_id = self.session_manager.get_or_create(
    user_id=message.author.id,           # Discord 사용자 ID
    username=message.author.name,        # 사용자 이름
    message_id=str(session_id)           # 스레드 ID (리플라이시)
)
```

**세션이란?**
- 사용자별 대화 기록을 유지하는 것
- 같은 사용자가 계속 대화하면 이전 메시지들을 기억함
- Gemini API의 `chat` 객체가 이것을 담당

**무슨 일이 일어나는가:**

```
먼저 확인: 이 사용자가 이미 세션이 있는가?
    ├─ YES → 기존 세션 재사용 (10분 이내)
    └─ NO → 새로운 세션 생성

새로운 세션을 만들 때:
    1. 데이터베이스에서 사용자 정보 로드
    2. 기존 기억들 불러오기
    3. 최근 대화 기록 불러오기
    4. 메모리 컨텍스트를 포함한 시스템 프롬프트 구성
    5. Gemini의 새로운 chat 세션 생성
```

**예시:**
```
첫 번째 메시지:
User: "소예봇 안녕"
→ 새로운 세션 생성
  - 메모리 로드: (없음)
  - 대화 기록: (없음)
  - Gemini chat 시작

같은 사용자의 2번째 메시지 (5분 후):
User: "소예봇 뭐해?"
→ 기존 세션 재사용
  - 이전 "안녕" 메시지 기억함
```

---

### 3단계: 메모리 컨텍스트 구성 (선택적)

**어디서:** `services/memory_service.py`

**메모리란?**
- 사용자가 입력한 정보들을 저장하는 것
- 나중에 대화에서 자동으로 참고됨

**어떤 메모리들이 있는가?**

```
1️⃣ Fact (사실)
   예: "사용자는 프로그래밍을 좋아한다"
   중요도: 0.7

2️⃣ Preference (선호도)
   예: "사용자는 짧은 답변을 원한다"
   중요도: 0.8

3️⃣ Key Memory (중요한 기억)
   예: "사용자의 생일은 12월 25일"
   중요도: 사용자 지정 (1-10)

4️⃣ Interaction Pattern (상호작용 패턴)
   예: 자주 나누는 주제, 감정 추이
```

**메모리를 불러오는 방법:**

```
방법 1: Inject All (권장 - 1GB RAM 환경)
├─ 최근 기억 20개를 모두 불러옴
├─ 빠르고 간단함
└─ 메모리 사용량 적음

방법 2: Semantic Search (의미론적 검색)
├─ 현재 대화와 유관한 기억만 찾음
├─ 더 정확함
└─ 리소스를 더 많이 사용 (모델 필요)
```

**처리 과정:**

```python
# 사용자의 모든 기억 불러오기
memories = self.memory_service.retrieve_memories(
    user_id=message.author.id,
    query=user_message,  # 현재 메시지와 관련된 것만
    max_memories=20
)

# 기억들을 텍스트로 포맷팅
memory_context = """
🧠 당신이 알고 있는 정보:

### Facts
- 사용자는 프로그래밍을 좋아한다
- 사용자는 대한민국에 산다

### Preferences
- 사용자는 짧은 답변을 원한다

### Key Memories
- 사용자의 생일은 12월 25일
"""
```

---

### 4단계: 메시지 저장 (기억에 남기기)

**어디서:** `bot/session.py` - `save_message()`

**무엇을 하는가:**
```python
# 사용자의 메시지를 데이터베이스에 저장
self.session_manager.save_message(
    user_id=message.author.id,
    session_id=str(session_id),
    role='user',           # 사용자가 보낸 메시지
    content=user_message   # "안녕하세요!"
)
```

**왜 저장하는가?**
- 나중에 대화 이력을 보기 위해
- 새로운 세션을 만들 때 이전 대화를 참고하기 위해
- 사용자의 상호작용 패턴을 분석하기 위해

---

### 5단계: AI 응답 생성

**어디서:** `services/gemini_service.py` - `generate_chat_response()`

**무엇을 하는가:**

```python
# Gemini API에 요청
response_text = await self.gemini_service.generate_chat_response(
    chat_session=chat_session,      # 사용자의 세션
    user_message=user_message,      # "안녕하세요!"
    progress_message=progress_msg,  # "생각 중... 💭"
    tools=tools                     # 함수 호출 도구 (선택)
)
```

**Gemini가 받는 정보:**

```
[시스템 프롬프트]
당신은 소예입니다. 무뚝뚝하고 인상을 쓰고 있지만...
(성격 설정)

[메모리 컨텍스트] ← 새로 추가됨!
🧠 당신이 알고 있는 정보:
- 사용자는 프로그래밍을 좋아한다
- 사용자는 대한민국에 산다

[대화 기록]
User: 안녕하세요!
```

**Gemini의 작업:**
```
1. 모든 정보를 읽음
2. "이 사용자에게 어떻게 대답할까?"를 고민
3. 응답 생성
4. 함수 호출이 필요하면 호출 (선택)
```

**함수 호출이란?**

Gemini가 자동으로 사용자의 정보를 기억할 수 있도록 도구를 제공:

```python
tools = [
    {
        "name": "save_user_fact",
        "description": "사용자에 대한 사실 저장",
        "parameters": {
            "fact": "사용자는 게임을 좋아한다",
            "category": "hobby"
        }
    },
    {
        "name": "save_preference",
        "description": "사용자의 선호도 저장",
        "parameters": {
            "preference": "사용자는 빠른 응답을 원한다"
        }
    },
    # ... 더 있음
]
```

**예시:**
```
User: "나는 게임 개발자야"

Gemini의 생각:
"아, 이 사용자는 게임을 좋아하고 개발자네!
save_user_fact를 호출해서 이 정보를 저장해야겠다"

→ 자동으로 호출:
save_user_fact(
    fact="사용자는 게임 개발자다",
    category="profession"
)

→ 기억 저장됨!
```

---

### 6단계: 기억 저장 처리

**어디서:** `bot/cogs/assistant.py` - `_handle_function_calls()`

**무엇을 하는가:**
```python
# Gemini가 호출한 함수들 처리
if 'save_user_fact' in response_text:
    logger.debug("사실 저장 함수가 호출됨")
    # 데이터베이스에 저장

if 'save_preference' in response_text:
    logger.debug("선호도 저장 함수가 호출됨")
    # 데이터베이스에 저장
```

**데이터베이스에 저장되는 것:**

```
테이블: memories
┌─────┬──────────┬────────────────────────┬────────────┐
│ id  │ user_id  │ memory_type            │ content    │
├─────┼──────────┼────────────────────────┼────────────┤
│ 1   │ 123456   │ fact                   │ 게임 개발자 │
│ 2   │ 123456   │ preference             │ 빠른 응답   │
└─────┴──────────┴────────────────────────┴────────────┘
```

---

### 7단계: 응답 저장

**어디서:** `bot/session.py` - `save_message()`

**무엇을 하는가:**
```python
# AI의 응답도 기억에 남기기
self.session_manager.save_message(
    user_id=message.author.id,
    session_id=str(session_id),
    role='assistant',      # AI가 보낸 메시지
    content=response_text  # "안녕하세요! 뭔데요?"
)
```

**왜 저장하는가?**
- 다음 세션을 시작할 때 대화 이력을 로드하기 위해
- "너가 이전에 뭐라고 했던 거 기억나?"하는 질문에 대응하기 위해

---

### 8단계: Discord에 응답 전송

**어디서:** `bot/cogs/assistant.py` - `on_message()`

**무엇을 하는가:**
```python
# 처음에 "생각 중... 💭"이라고 했던 메시지를 업데이트
await DiscordUI.safe_edit(progress_message, response_text)

# 또는 새로운 메시지로 전송
await message.reply(response_text)
```

**예시:**
```
[처음]
Bot: "생각 중... 💭"

[응답 생성 후]
Bot: "안녕하세요! 뭔데요?"
```

---

## 📋 전체 흐름 요약

```
1️⃣ 사용자 메시지 도착
   └─ "@소예봇 안녕"

2️⃣ 봇이 @mention 감지
   └─ "내가 불렸네"

3️⃣ 사용자 세션 생성/재사용
   └─ 기존 기억과 대화 이력 로드

4️⃣ 메모리 컨텍스트 구성
   └─ "이 사용자는 이런 사람이야"

5️⃣ 사용자 메시지 저장
   └─ 데이터베이스에 기록

6️⃣ Gemini API에 요청 (메모리 포함)
   └─ "이 정보로 응답해줘"

7️⃣ Gemini 응답 생성
   └─ 함수 호출 가능

8️⃣ 함수 호출 처리 (기억 저장)
   └─ "아, 이 정보를 저장해야겠다"

9️⃣ AI 응답 저장
   └─ 데이터베이스에 기록

🔟 Discord에 응답 전송
   └─ 사용자가 읽음
```

---

## 🗂️ 주요 파일 역할

| 파일 | 역할 | 주요 기능 |
|------|------|---------|
| `main.py` | 봇의 시작점 | 모든 서비스 초기화, Cog 로드 |
| `bot/cogs/assistant.py` | 메시지 수신 및 처리 | @mention 감지, 응답 생성 |
| `bot/cogs/memory.py` | 기억 관리 명령어 | !기억 저장, 조회, 삭제 |
| `bot/session.py` | 사용자 세션 관리 | 사용자별 대화 세션 유지 |
| `services/gemini_service.py` | AI 서비스 | Gemini API 호출 |
| `services/memory_service.py` | 기억 처리 | 기억 저장, 검색, 로드 |
| `services/database_service.py` | 데이터베이스 | SQLite 메모리 저장 |
| `prompts.py` | 성격 설정 | 소예 캐릭터 프롬프트 |

---

## 🔌 데이터 흐름도

```
┌─────────────────────────────────────┐
│    Discord User                     │
│    "@소예봇 안녕"                   │
└────────────────┬────────────────────┘
                 │ (메시지)
                 ▼
         ┌───────────────┐
         │ AssistantCog  │ (수신, 처리)
         └───────┬───────┘
                 │
         ┌───────▼──────────────────┐
         │  SessionManager          │
         │  (세션 생성/재사용)       │
         └───────┬──────────────────┘
                 │
        ┌────────┴─────────────────────┐
        │                              │
    ┌───▼─────┐              ┌────────▼──────┐
    │ Memory  │              │ Database      │
    │ Service │              │ Service       │
    │ (기억)   │              │ (저장)        │
    └───┬─────┘              └────────┬──────┘
        │                             │
        └──────────┬──────────────────┘
                   │
            ┌──────▼───────┐
            │ Gemini API   │ (AI 응답)
            └──────┬───────┘
                   │
        ┌──────────▼──────────┐
        │ Function Calling    │
        │ (기억 저장)          │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Discord User        │
        │ "안녕하세요!"        │
        └─────────────────────┘
```

---

## 💾 메모리 저장소 구조

```
SQLite Database (soyebot.db)
│
├─ users 테이블
│  ├─ user_id (Discord ID)
│  ├─ username
│  ├─ created_at
│  └─ config_json
│
├─ memories 테이블
│  ├─ id
│  ├─ user_id
│  ├─ memory_type (fact, preference, key_memory)
│  ├─ content
│  ├─ importance_score
│  └─ embedding (의미 검색용)
│
├─ conversation_history 테이블
│  ├─ id
│  ├─ user_id
│  ├─ role (user, assistant)
│  ├─ content
│  └─ timestamp
│
└─ interaction_patterns 테이블
   ├─ user_id
   ├─ total_messages
   ├─ favorite_topics
   └─ sentiment_avg
```

---

## ⚙️ 주요 설정값

```python
# config.py
memory_model_name = 'gemini-1.5-flash-8b'  # 기억용 경량 모델
memory_retrieval_mode = 'inject_all'       # 기억 검색 방식
max_memories_to_inject = 20                # 한 번에 로드할 기억 수
max_conversation_history = 50              # 저장할 대화 메시지 수
memory_compression_days = 7                # 오래된 대화 압축 기간
enable_memory_system = True                # 기억 시스템 활성화
```

---

## 🚀 실제 사용 예시

### 예시 1: 첫 대화

```
User: "@소예봇 안녕! 난 프로그래밍을 좋아해"

처리 과정:
1. 봇이 @mention 감지
2. 새 세션 생성 (기억 없음)
3. Gemini에게 전달:
   - 시스템 프롬프트 (성격)
   - 유저 메시지

4. Gemini 응답:
   "뭐하는 건데. 프로그래밍?"
   (함수 호출: save_user_fact("프로그래밍 좋아함", "hobby"))

5. 기억 저장: "사용자는 프로그래밍을 좋아한다"
6. Discord에 응답 전송

Database:
├─ User 생성
├─ Memory 저장: "프로그래밍 좋아함"
└─ Conversation 저장: 사용자 메시지 + AI 응답
```

### 예시 2: 두 번째 대화 (같은 사용자, 5분 후)

```
User: "@소예봇 뭐 했어?"

처리 과정:
1. 봇이 @mention 감지
2. 기존 세션 재사용
3. 메모리 로드: "프로그래밍 좋아함"
4. 대화 이력 로드: 이전 "안녕! 난 프로그래밍을 좋아해" 기억함
5. Gemini에게 전달:
   - 시스템 프롬프트
   - 메모리 (프로그래밍 좋아함)
   - 대화 이력 (이전 대화)
   - 새 메시지

6. Gemini: "프로그래밍 하면서 뭐했지?"
   (기억이 있어서 더 맥락있는 응답)

7. Discord에 응답 전송

Database:
└─ Conversation 추가: 새 메시지 + 새 응답
```

---

## 🎯 핵심 개념 3가지

### 1️⃣ Session (세션)
- **뜻:** 사용자와의 대화 세션
- **역할:** 같은 사용자의 메시지들을 연결
- **기간:** 10분 동안 활동이 없으면 만료
- **이점:** 대화 이력을 자동으로 기억

### 2️⃣ Memory (기억)
- **뜻:** 사용자에 대해 학습한 정보
- **종류:** 사실, 선호도, 중요한 기억
- **저장:** 데이터베이스에 영구 저장
- **용도:** 미래 대화에서 참고

### 3️⃣ Context (컨텍스트)
- **뜻:** Gemini에게 제공하는 모든 정보
- **포함:** 성격, 기억, 대화 이력
- **목적:** AI가 더 정확하고 맥락있는 응답을 하도록 함
- **형태:** 텍스트로 변환되어 프롬프트에 포함됨

---

## 📈 데이터 흐름 예시 (시간순)

```
시간 0: User가 "@소예봇 안녕!"이라고 말함
├─ AssistantCog.on_message() 호출
├─ SessionManager.get_or_create() 호출
├─ 새 세션 생성
└─ Database: User 생성, Conversation 저장

시간 1: Gemini API 호출 (기억 없음)
├─ 요청: "안녕!"
└─ 응답: "뭔데요. 인사하냐고?"

시간 2: 응답 저장
├─ Database: Conversation에 AI 응답 저장
└─ Discord에 전송

시간 5분: 같은 User가 "@소예봇 뭐해?"라고 말함
├─ AssistantCog.on_message() 호출
├─ SessionManager.get_or_create() 호출
├─ 기존 세션 찾음 (10분 이내)
├─ 메모리 로드 (있으면)
├─ 대화 이력 로드 (5분 전 대화)
└─ Database: 대화 이력 조회

시간 6분: Gemini API 호출 (기억 + 대화 이력 포함)
├─ 요청: "뭐해?" + 이전 대화
└─ 응답: "아까 인사했으니까 또 뭐 하려고"

시간 7분: 응답 저장
├─ Database: Conversation에 AI 응답 저장
└─ Discord에 전송
```

---

## 🔍 정리

**한 줄 요약:**
> **사용자가 메시지 → 기억/세션 로드 → Gemini 호출 → 기억 저장 → 응답 전송**

**세부 단계:**
1. 메시지 수신 감지
2. 사용자 세션 관리
3. 메모리 로드
4. Gemini API 호출
5. 함수 호출로 기억 저장
6. 응답 및 메시지 저장
7. Discord에 응답 전송

**이 구조의 장점:**
- ✅ 사용자별 개인화된 대화
- ✅ 긴 대화 이력 기억
- ✅ 자동 정보 추출 및 저장
- ✅ 1GB RAM 환경에서 작동 (최적화됨)
