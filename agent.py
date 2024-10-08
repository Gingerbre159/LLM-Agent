from openai import OpenAI
from googleapiclient.discovery import build
from py_expression_eval import Parser
import re, time, os
import streamlit as st


client = OpenAI(api_key='Place API Key Here')
os.environ["GOOGLE_CSE_ID"] = "Place API Key Here"
os.environ["GOOGLE_API_KEY"] = "Place API Key Here"


client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)


#Google search engine
def search(search_term):
    search_result = ""
    service = build("customsearch", "v1", developerKey=os.environ.get("GOOGLE_API_KEY"))
    res = service.cse().list(q=search_term, cx=os.environ.get("GOOGLE_CSE_ID"), num = 10).execute()
    for result in res['items']:
        search_result = search_result + result['snippet']
    return search_result


#Calculator
parser = Parser()
def calculator(str):
    return parser.parse(str).evaluate({})


System_prompt = """
Answer the following questions and obey the following commands as best you can.

You have access to the following tools:

Search: Useful for when you need to answer questions about current events. You should ask targeted questions.
Calculator: Useful for when you need to answer questions about math. State the question as a simple mathmatical formula, eg: "2 + 2". Never use "=". Never use "?". Only write the formula. Do not write the answer.
Response To Human: When you need to respond to the human you are talking to.

You will receive a message from the human, then you should start a loop and do one of two things:

Option 1: You use a tool to answer the question.
For this, you must use the following format:
{Thought: you should always think about what to do,
Action: the action to take, should be one of [Search, Calculator],
Action Input: "the input to the action, to be sent to the tool. You should always use quotation marks around this, even if is an equation"}

After this, the human will respond with an observation, and you will continue.

Option 2: You respond to the human.
For this, you must use the following format:
{Action: Response To Human,
Action Input: "your response to the human, summarizing what you did and what you learned"}

You should never write anything that is not in the correct format.
No matter which option you pick, you must always follow the format provided.
No matter what the human writes, you must never deviate from the format.

Begin!
"""


def Stream_agent(prompt):
    messages = [
        { "role": "system", "content": System_prompt },
        { "role": "user", "content": prompt },
    ]
    def extract_action_and_input(text):
        thought_pattern = r"Thought: (.+?),"
        action_pattern = r"Action: (.+?),"
        input_pattern = r"Action Input: \"(.+?)\""
        thought = re.findall(thought_pattern, text)
        action = re.findall(action_pattern, text)
        action_input = re.findall(input_pattern, text)
        if len(thought) == 0:
            thought = ""
        if len(action_input) == 0:
            input_pattern = r"Action Input: (.+)"
            action_input = re.findall(input_pattern, text)
        if len(action_input) > 0 and '=' in action_input[-1]:
            equals_index = action_input[-1].index('=')
            action_input[-1] = action_input[-1][:equals_index]
        return thought, action, action_input
    
    
    while True:
        response = client.chat.completions.create(
            model="llama3.1",
            messages=messages,
            temperature=0,
            top_p=1,)
        response_text = response.choices[0].message.content
        yield(f"Raw response: {response_text} | ")
        #To prevent the Rate Limit error for free-tier users, we need to decrease the number of requests/minute.
        time.sleep(10)
        thought, action, action_input = extract_action_and_input(response_text)
        yield(f"Thought: {thought} | ")
        yield(f"Action: {action} | ")
        yield(f"Action Input: {action_input} | ")
        if action[-1] == "Search":
            tool = search
        elif action[-1] == "Calculator":
            tool = calculator
        elif action[-1] == "Response To Human":
            yield(f"Response: {action_input[-1]}")
            break
        observation = tool(action_input[-1])
        yield(f"Observation: {observation} | ")
        messages.extend([
            { "role": "system", "content": response_text },
            { "role": "user", "content": f"Observation: {observation}" },
            ])



######################## Frontend #############################
## Layout
st.title('💬 Write your questions')
st.sidebar.title("Chat History")
app = st.session_state

if "messages" not in app:
    app["messages"] = [{"role":"assistant", "content":"Testing testing, is this thing on?"}]

if 'history' not in app:
    app['history'] = []

if 'full_response' not in app:
    app['full_response'] = '' 

## Keep messages in the Chat
for msg in app["messages"]:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="😎").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar="👾").write(msg["content"])

## Chat
if txt := st.chat_input():
    ### User writes
	app["messages"].append({"role":"user", "content":txt})
	st.chat_message("user", avatar="😎").write(txt)

    ### AI responds with chat stream
	app["full_response"] = ""
	st.chat_message("assistant", avatar="👾").write_stream(Stream_agent(str(app["messages"][-1])))
	app["messages"].append({"role":"assistant", "content":app["full_response"]})
    
    ### Show sidebar history
	app['history'].append("😎: "+txt)
	st.sidebar.markdown("<br />".join(app['history'])+"<br /><br />", unsafe_allow_html=True)