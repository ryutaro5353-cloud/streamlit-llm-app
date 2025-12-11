# app.py
import os

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
#  基本設定
# =========================
st.set_page_config(
    page_title="LangChain × Streamlit LLMデモ",
    page_icon="🤖",
)

st.title("🤖 LangChain × Streamlit LLMデモアプリ")

st.write(
    """
### アプリの概要

このアプリでは、次のことができます。

1. **専門家の種類（A/B）** をラジオボタンで選択  
2. 下のテキストボックスに質問や相談内容を入力  
3. 「送信」ボタンを押すと、選んだ専門家として LLM が回答を生成  

---

**使い方：**

1. まず、上のラジオボタンから「どんな専門家として答えてほしいか」を選びます  
2. 次に、下のテキスト入力欄に聞きたい内容を書きます  
3. 「送信」ボタンを押すと、画面下部に AI からの回答が表示されます  
"""
)

# =========================
#  OpenAI / LLM の準備
# =========================
# 環境変数から API キーを取得（Streamlit Cloud では Secrets に設定推奨）
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.warning(
        "環境変数 `OPENAI_API_KEY` が設定されていません。"
        "ローカルでは `.env` やシェルに設定し、"
        "Streamlit Community Cloud では Secrets に設定してください。"
    )

# LangChain の LLM インスタンス
# model はお好みで変更可（例: "gpt-4o-mini" など）
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)


# =========================
#  LLM 呼び出し用の関数
# =========================
def call_llm(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家の種類をもとに、
    LangChain 経由で LLM に問い合わせを行い、回答テキストを返す関数。

    Args:
        user_input (str): ユーザーが入力したテキスト
        expert_type (str): ラジオボタンで選択された専門家の種類

    Returns:
        str: LLM からの回答テキスト
    """

    # 専門家ごとに system メッセージを変更
    if expert_type == "栄養・食事の専門家（A）":
        system_message = (
            "あなたはプロの管理栄養士・栄養学の専門家です。"
            "ユーザーの健康状態や生活スタイルを想像しつつ、"
            "分かりやすく、具体的な食事アドバイスを日本語で行ってください。"
        )
    elif expert_type == "旅行プランナーの専門家（B）":
        system_message = (
            "あなたは世界中の観光地・フライト・ホテル事情に詳しい、"
            "プロの旅行プランナーです。ユーザーの希望を踏まえて、"
            "現実的でワクワクする旅程やプランを日本語で提案してください。"
        )
    else:
        # デフォルト（想定外の値が来たとき）
        system_message = (
            "あなたは丁寧で分かりやすい日本語で回答する、"
            "汎用的なAIアシスタントです。"
        )

    # Lesson8 を意識した ChatPromptTemplate + LCEL 形式
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{user_input}"),
        ]
    )

    # チェーン（Prompt → LLM → 出力パーサ）
    chain = prompt | llm | StrOutputParser()

    # LangChain で LLM 呼び出し
    response_text = chain.invoke({"user_input": user_input})

    return response_text


# =========================
#  Streamlit UI 部分
# =========================

# 1. 専門家ロールの選択（ラジオボタン）
expert_type = st.radio(
    "LLM にどんな専門家として振る舞ってほしいですか？",
    options=[
        "栄養・食事の専門家（A）",
        "旅行プランナーの専門家（B）",
    ],
    horizontal=False,
)

# 2. ユーザーの入力テキスト
user_input = st.text_area(
    "質問や相談内容を入力してください：",
    height=150,
    placeholder="例）筋トレしているのですが、1日のタンパク質量の目安を教えてください。\n"
                "例）年末年始に3泊4日で行ける海外旅行プランを提案してください。",
)

# 3. 送信ボタン
submitted = st.button("送信")

# 4. 送信が押されたら LLM を呼び出して結果を表示
if submitted:
    if not user_input.strip():
        st.error("テキストが未入力です。何か質問や相談内容を入力してください。")
    else:
        with st.spinner("LLM に問い合わせ中です…"):
            answer = call_llm(user_input=user_input, expert_type=expert_type)

        st.markdown("### 🧠 LLM からの回答")
        st.write(answer)
