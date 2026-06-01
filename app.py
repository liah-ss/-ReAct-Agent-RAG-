"""
智扫通 - 扫地机器人智能客服系统
基于 ReAct Agent + RAG 的智能问答系统
"""
import os
import streamlit as st
from agent.react_agent import ReactAgent

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="智扫通 - 智能客服",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS 样式 ====================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
h1 { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
div.stChatInput > div {
    border-radius: 25px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}
div.stButton > button {
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    border: 1px solid #e0e0e0 !important;
}
div.stButton > button:hover {
    border-color: #667eea !important;
    box-shadow: 0 4px 12px rgba(102,126,234,0.3) !important;
}
.status-dot {
    display:inline-block; width:8px; height:8px;
    background:#22c55e; border-radius:50%; margin-right:6px;
    animation:pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
</style>
""", unsafe_allow_html=True)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:3rem;'>🤖</div>
        <div style='font-size:1.3rem;font-weight:bold;color:#667eea;'>智扫通</div>
        <div style='font-size:0.8rem;color:#888;'>智能客服系统</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🎯 核心功能")
    cards = [
        ("🧠", "ReAct 思考", "#e0eafc,#cfdef3"),
        ("📚", "RAG 检索", "#f5f7fa,#c3cfe2"),
        ("📊", "报告生成", "#fff5f5,#ffe4e6"),
        ("🌤️", "天气适配", "#f0fdf4,#bbf7d0"),
    ]
    for emoji, label, colors in cards:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,{colors});"
            f"padding:0.8rem;border-radius:12px;margin-bottom:0.5rem;font-size:0.9rem;font-weight:500;'>"
            f"<span style='font-size:1.3rem;margin-right:0.5rem;'>{emoji}</span>{label}</div>",
            unsafe_allow_html=True
        )
    st.divider()

    st.markdown("### 📖 知识库")
    for item in ["🏠 扫地机器人100问", "🧹 扫拖一体指南", "🔧 维护保养", "🛒 选购指南", "⚠️ 故障排除"]:
        st.markdown(f"<div style='padding:0.3rem 0;color:#555;'>{item}</div>", unsafe_allow_html=True)
    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.markdown("<div style='text-align:center;color:#888;font-size:0.75rem;'>✨ 智扫通 v2.0<br>🚀 Powered by LangChain + RAG</div>", unsafe_allow_html=True)

# ==================== 主界面 ====================
col1, col2, col3, col4 = st.columns([1, 3, 3, 3])
with col1:
    st.markdown('<span class="status-dot"></span><span style="color:#22c55e;font-weight:500;">在线</span>', unsafe_allow_html=True)
with col2: st.caption("📚 知识库已加载")
with col3: st.caption("🤖 Agent 已就绪")
with col4: st.caption("🟢 系统正常")
st.divider()

st.markdown("""
<div style='text-align:center; padding:1.5rem 0 1rem 0;'>
    <div style='font-size:3rem;'>🤖</div>
    <h1 style='margin:0; font-size:2.5rem; font-weight:700;'>智扫通</h1>
    <p style='color:#888; margin-top:0.5rem; font-size:1rem;'>扫地机器人智能客服 · 7×24 小时为您服务</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ==================== 初始化 ====================
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "agent_error" not in st.session_state:
    st.session_state["agent_error"] = None

# 初始化 Agent（带详细错误提示）
if st.session_state["agent"] is None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        st.error("""
        ### ❌ 缺少 API 密钥
        请在 `.env` 文件中配置 `DASHSCOPE_API_KEY`，或设置环境变量。
        """)
        st.code("export DASHSCOKE_API_KEY=your-api-key-here", language="bash")
        st.stop()
    try:
        with st.spinner("🚀 正在初始化智能客服..."):
            st.session_state["agent"] = ReactAgent()
        st.success("✅ 智能客服初始化成功！")
    except Exception as e:
        st.session_state["agent_error"] = str(e)
        st.error(f"""
        ### ❌ 初始化失败
        原因：{str(e)}
        """)
        st.info("💡 请确保 API 密钥有效且网络连接正常")
        st.stop()

# 如果之前有错误，显示并停止
if st.session_state.get("agent_error"):
    st.error(f"⚠️ Agent 发生错误：{st.session_state['agent_error']}")
    st.stop()

# ==================== 处理消息的公共函数 ====================
def handle_message(msg: str):
    """处理用户消息并生成回复（流式输出）"""
    # 先保存用户消息
    st.session_state["chat_history"].append({"role": "user", "content": msg})

    # 显示用户消息
    with st.chat_message("user", avatar="👤"):
        st.markdown(msg)

    # 流式生成 AI 回复
    with st.chat_message("assistant", avatar="🤖"):
        try:
            # st.write_stream 消费生成器，实时显示，并返回完整结果
            full_response = st.write_stream(
                st.session_state["agent"].execute_stream(msg)
            )

            if full_response:
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": full_response
                })
            else:
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": "抱歉，暂时无法回答您的问题，请稍后再试。"
                })
        except Exception as e:
            st.error(f" 发生错误：{str(e)}")
            st.info("请检查您的 API 密钥是否配置正确，以及网络连接是否正常。")
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": f"抱歉，处理您的请求时发生了错误：{str(e)}"
            })

    st.rerun()

# ==================== 欢迎界面（仅首次访问） ====================
if len(st.session_state["chat_history"]) == 0:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#faf5ff 0%,#f0f9ff 100%);
                border-radius:20px; padding:2rem; text-align:center; margin:1rem 0;
                border:1px solid #e0e0e0;'>
        <div style='font-size:3rem;'>👋</div>
        <h2 style='color:#333;'>欢迎来到智扫通智能客服！</h2>
        <p style='color:#666;'>我是您的扫地机器人专属智能助手，请选择您想了解的问题：</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 快速提问")
    questions = [
        "🏠 小户型适合哪些扫地机器人？",
        "🔋 扫地机器人电池如何保养？",
        "🧹 扫拖一体机和纯扫地怎么选？",
        "⚠️ 机器人经常迷路怎么办？",
        "💰 2000元预算推荐哪款？",
        "📋 生成我的使用报告",
    ]
    cols = st.columns(3)
    for i, q in enumerate(questions):
        with cols[i % 3]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                handle_message(q)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1: st.info("**🔍 知识问答**\n基于专业知识库，解答选购/使用/维护问题")
    with c2: st.info("**📊 报告生成**\n查询使用记录，生成个性化报告")
    with c3: st.info("**🌐 智能建议**\n结合环境因素给出使用建议")

# ==================== 聊天区域 ====================
for msg in st.session_state["chat_history"]:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])

# 聊天输入框
prompt = st.chat_input('💬 输入您的问题，例如："小户型适合哪些扫地机器人？"')

if prompt:
    handle_message(prompt)

# 页脚
st.markdown("---")
st.markdown("<div style='text-align:center;color:#888;font-size:0.75rem;padding:0.5rem 0;'>© 2026 智扫通智能客服</div>", unsafe_allow_html=True)
