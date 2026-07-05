import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import base64
import json

# ====================== 页面基础配置 ======================
st.set_page_config(page_title="微信风格客服", page_icon="💬", layout="wide")

# ====================== API客户端初始化 ======================
# 校验密钥是否配置
if "DOUBAO_API_KEY" not in st.secrets:
    st.error("❌ 请在后台 secrets.toml 配置 DOUBAO_API_KEY = '你的sk-密钥'")
    st.stop()

client = OpenAI(
    api_key=st.secrets["DOUBAO_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

# -------------------------- 重要替换区域 --------------------------
# 1. 纯文字模型（DeepSeek-V4-flash，支持tools工具调用，填入你真实ep-开头ID）
TEXT_MODEL = "ep-20260705195949-pr84t"
# 2. 视觉识图模型（你原有可用的视觉接入ID，无需修改）
VISION_MODEL = "ep-20260705180241-s57gl"
# -----------------------------------------------------------------

# ====================== 店铺静态知识库 ======================
KB_DATA = [
    "营业时间：10:00-23:00",
    "预约电话：13812345678",
    "3个包间，提前1天预订",
    "人均120元",
    "招牌菜：酸菜鱼、麻辣香锅、蒜蓉小龙虾",
    "会员充值500送50，1000送150",
    "每周二会员日8折",
    "生日当天凭身份证送招牌菜一份"
]

def search_kb(query, top_k=2):
    """简易关键词检索知识库，无向量库依赖"""
    score_list = []
    for text in KB_DATA:
        match_count = sum(1 for word in query if word in text)
        score_list.append((match_count, text))
    # 匹配度从高到低排序
    score_list.sort(reverse=True, key=lambda x: x[0])
    return [text for score, text in score_list[:top_k] if score > 0]

# ====================== 订单模拟数据库 & 工具函数 ======================
orders_db = {
    "001": {"customer": "张三", "item": "酸菜鱼", "status": "配送中", "phone": "13900001111"},
    "002": {"customer": "李四", "item": "麻辣香锅", "status": "已签收", "phone": "13900002222"},
}

def query_order(order_id: str):
    if order_id in orders_db:
        info = orders_db[order_id]
        return f"订单{order_id}：顾客{info['customer']}，菜品{info['item']}，当前状态：{info['status']}"
    return "未查询到该订单，请核对订单编号"

def refund_order(order_id: str):
    if order_id in orders_db:
        orders_db[order_id]["status"] = "已退款"
        return f"订单{order_id}退款办理成功"
    return "退款失败，无此订单编号"

# 工具定义（仅TEXT_MODEL文字模型调用）
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询顾客订单配送状态，必须传入订单号",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "用户提供的订单编号"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "为指定订单执行退款操作，需要订单编号",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "需要退款的订单编号"}
                },
                "required": ["order_id"]
            }
        }
    }
]

def execute_tool(tool_name: str, args: dict):
    """执行工具，捕获异常避免崩溃"""
    try:
        if tool_name == "query_order":
            return query_order(args["order_id"])
        elif tool_name == "refund_order":
            return refund_order(args["order_id"])
        else:
            return f"未知工具：{tool_name}"
    except Exception as e:
        return f"工具执行异常：{str(e)}"

# ====================== 图片压缩编码函数 ======================
def encode_image(upload_file, max_edge=1024):
    img = Image.open(upload_file)
    w, h = img.size
    # 长边压缩至1024以内
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # 转base64
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_str

# ====================== 视觉模型专用：图片识别 ======================
def vision_analyze(image_b64: str, prompt_text: str) -> str:
    """仅调用视觉模型，纯看图，不涉及工具调用"""
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return resp.choices[0].message.content
    except Exception as err:
        return f"图片识别失败：{str(err)}"

# ====================== 文字Agent对话核心（支持tools工具调用） ======================
def text_agent_chat(user_input: str, history_msg_list: list):
    # 检索知识库补充上下文
    kb_match_text = "\n".join(search_kb(user_input)) or "暂无相关门店信息"
    system_prompt = f"""你是线下实体店亲切客服，说话口语化，多用亲、呢、哦，回复控制30~50字。
门店基础信息：
{kb_match_text}

规则：
1. 用户询问订单、退款，自动调用对应工具；
2. 不清楚的信息不要编造，如实告知；
3. 顾客上传菜品图片，根据图片描述写适合门店的朋友圈宣传文案。"""

    # 组装完整对话上下文
    msg_list = [{"role": "system", "content": system_prompt}] + history_msg_list
    msg_list.append({"role": "user", "content": user_input})

    try:
        # 第一次请求：判断是否需要调用工具
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=msg_list,
            tools=tool_definitions,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500
        )
        ai_msg = response.choices[0].message

        # 存在工具调用，执行工具并二次请求生成最终回答
        if ai_msg.tool_calls:
            tool_call_info = ai_msg.tool_calls[0]
            # 解析工具参数
            tool_args = json.loads(tool_call_info.function.arguments)
            tool_result_content = execute_tool(tool_call_info.function.name, tool_args)

            # 把工具调用记录写入上下文，保证多轮对话记忆
            msg_list.append(ai_msg)
            msg_list.append({
                "role": "tool",
                "tool_call_id": tool_call_info.id,
                "content": tool_result_content
            })

            # 二次请求，生成带工具结果的客服回复
            final_response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=msg_list,
                temperature=0.7,
                max_tokens=300
            )
            final_text = final_response.choices[0].message.content
            # 返回：回答文本、更新后的干净对话历史、工具日志
            return final_text, msg_list[1:], {"tool": tool_call_info.function.name, "result": tool_result_content}
        else:
            # 无工具调用，直接返回
            return ai_msg.content, msg_list[1:], None

    except Exception as err:
        return f"❌ 系统异常：{str(err)}", history_msg_list, None

# ====================== Session状态初始化（仅首次运行执行，无重复） ======================
if "chat_history" not in st.session_state:
    # 传给模型的纯文本对话历史，干净无多余UI字段
    st.session_state.chat_history = []

if "ui_messages" not in st.session_state:
    # 页面渲染用消息，支持图片、工具日志展示
    st.session_state.ui_messages = [
        {
            "role": "assistant",
            "content": "亲，你好呀！👋\n我是店小秘，有什么可以帮你的吗？\n\n你可以问我：\n• 营业时间、菜单价格\n• 订单到哪里了\n• 发张照片我帮你写文案"
        }
    ]

if "last_handle_img_id" not in st.session_state:
    # 图片去重标记，防止上传后无限循环刷新
    st.session_state.last_handle_img_id = None

# ====================== 页面UI渲染（仿微信气泡） ======================
st.markdown("""
<div style="background-color:#07C160; padding:10px; border-radius:10px 10px 0 0; text-align:center;">
    <h3 style="color:white; margin:0;">💬 店铺客服</h3>
    <small style="color:#E8F5E9;">在线 · 秒回</small>
</div>
""", unsafe_allow_html=True)

# 聊天消息容器
chat_box = st.container()
with chat_box:
    for single_msg in st.session_state.ui_messages:
        if single_msg["role"] == "user":
            # 用户右侧绿色气泡
            display_content = single_msg["content"]
            # 附带图片则渲染图片
            if "image" in single_msg:
                display_content = f'<img src="{single_msg["image"]}" style="max-width:200px; border-radius:8px;"><br/>' + display_content
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin:8px 0;">
                <div style="background:#95EC69; padding:10px 15px; border-radius:15px 5px 15px 15px; max-width:70%; word-break:break-all;">
                    <small>{display_content}</small>
                </div>
                <div style="width:35px;height:35px;background:#07C160;border-radius:50%;margin-left:8px;display:flex;align-items:center;justify-content:center;color:#fff;flex-shrink:0;">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # AI左侧白色气泡
            display_content = single_msg["content"]
            if "tool_log" in single_msg:
                display_content += f"\n\n🔧 执行工具：{single_msg['tool_log']['tool']}"
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-start; margin:8px 0;">
                <div style="width:35px;height:35px;background:#07C160;border-radius:50%;margin-right:8px;display:flex;align-items:center;justify-content:center;color:#fff;flex-shrink:0;">🤖</div>
                <div style="background:#fff; padding:10px 15px; border:1px solid #eee; border-radius:5px 15px 15px 15px; max-width:70%; white-space:pre-wrap; word-break:break-all;">
                    <small>{display_content}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

# 底部输入栏分割线
st.divider()
col_upload, col_input, col_reset = st.columns([1, 6, 1])

with col_upload:
    upload_img_file = st.file_uploader("📷", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="img_upload")

with col_input:
    user_text_input = st.chat_input("输入消息...", key="chat_text_input")

with col_reset:
    if st.button("🔄 清空对话", help="重置全部聊天记录"):
        st.session_state.chat_history = []
        st.session_state.ui_messages = [
            {"role": "assistant", "content": "亲，你好呀！👋\n我是店小秘，有什么可以帮你的吗？"}
        ]
        st.session_state.last_handle_img_id = None
        st.rerun()

# ====================== 文本消息处理逻辑 ======================
if user_text_input:
    # 1. 页面追加用户消息，仅执行一次
    st.session_state.ui_messages.append({"role": "user", "content": user_text_input})

    with st.spinner("AI正在回复中..."):
        reply_text, updated_history, tool_log_data = text_agent_chat(user_text_input, st.session_state.chat_history)

    # 2. 更新模型对话历史（只追加一轮用户+一轮AI，无重复）
    st.session_state.chat_history = updated_history
    st.session_state.chat_history.append({"role": "user", "content": user_text_input})
    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})

    # 3. UI追加AI回复
    ai_msg_item = {"role": "assistant", "content": reply_text}
    if tool_log_data:
        ai_msg_item["tool_log"] = tool_log_data
    st.session_state.ui_messages.append(ai_msg_item)
    st.rerun()

# ====================== 上传图片处理逻辑（防无限循环） ======================
if upload_img_file:
    # 生成图片唯一标识：文件名+文件大小，区分不同图片
    current_img_tag = f"{upload_img_file.name}_{upload_img_file.size}"
    # 仅未处理过的图片才执行识别
    if current_img_tag != st.session_state.last_handle_img_id:
        # 图片转base64
        img_b64 = encode_image(upload_img_file)
        img_data_url = f"data:image/jpeg;base64,{img_b64}"
        # UI新增用户图片消息
        st.session_state.ui_messages.append({
            "role": "user",
            "content": "帮我看看这张菜品，写一条门店朋友圈文案",
            "image": img_data_url
        })

        with st.spinner("识别菜品图片中..."):
            # 第一步：视觉模型解析图片内容
            img_analysis_result = vision_analyze(
                img_b64,
                "识别图片里的菜品，描述口感、外观，生成适合实体店宣传的朋友圈文案，40-80字，带emoji表情"
            )
            # 第二步：把图片识别结果交给文字Agent整合客服话术
            agent_prompt = f"用户上传了一张菜品图片，图片识别结果：{img_analysis_result}，请整理成友好的客服回复发给用户"
            reply_text, updated_history, tool_log_data = text_agent_chat(agent_prompt, st.session_state.chat_history)

        # 更新对话历史，仅追加一次，无重复
        st.session_state.chat_history = updated_history
        st.session_state.chat_history.append({"role": "user", "content": f"[图片消息] {img_analysis_result}"})
        st.session_state.chat_history.append({"role": "assistant", "content": reply_text})

        # UI添加AI回复气泡
        ai_msg_item = {"role": "assistant", "content": reply_text}
        if tool_log_data:
            ai_msg_item["tool_log"] = tool_log_data
        st.session_state.ui_messages.append(ai_msg_item)

        # 标记该图片已处理，刷新后不再重复执行
        st.session_state.last_handle_img_id = current_img_tag
        st.rerun()
