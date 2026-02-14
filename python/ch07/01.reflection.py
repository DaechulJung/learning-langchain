from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

model = ChatOpenAI(model="gpt-4o-mini")

# 프롬프트 정의
generate_prompt = SystemMessage(
    content=(
        "당신은 훌륭한 3단락 에세이를 작성하는 임무를 가진 에세이 어시스턴트입니다."
        " 사용자의 요청에 맞춰 최상의 에세이를 작성하세요."
        " 사용자가 비평을 제공하면, 이전 시도에 대한 수정 버전을 응답하세요."
    )
)

reflection_prompt = SystemMessage(
    content=(
        "당신은 에세이 제출물을 평가하는 교사입니다."
        " 사용자의 제출물에 대해 비평과 추천을 생성하세요."
        " 길이, 깊이, 스타일 등과 같은 구체적인 요구사항을 포함한 자세한 추천을 제공하세요."
    )
)


def generate(state: MessagesState) -> MessagesState:
    answer = model.invoke([generate_prompt, *state["messages"]])
    return {"messages": [answer]}


def _invert_message_role(msg: BaseMessage) -> BaseMessage:
    """Reflection 단계에서 assistant/user 역할을 뒤집어 self-critique를 유도합니다."""
    if isinstance(msg, AIMessage):
        return HumanMessage(content=msg.content)
    if isinstance(msg, HumanMessage):
        return AIMessage(content=msg.content)
    return msg


def reflect(state: MessagesState) -> MessagesState:
    # 첫 메시지는 원래 사용자 요청이므로 그대로 두고,
    # 이후 메시지만 반전하여 모델이 자신의 답변을 비평하도록 구성합니다.
    translated_messages = [
        reflection_prompt,
        state["messages"][0],
        *(_invert_message_role(msg) for msg in state["messages"][1:]),
    ]

    answer = model.invoke(translated_messages)
    # 비평 결과를 다음 생성 단계의 사용자 피드백으로 전달합니다.
    return {"messages": [HumanMessage(content=answer.content)]}


def should_continue(state: MessagesState) -> Literal["reflect", END]:
    # 3회 반복 후 종료 (초기 요청 1개 + generate/reflect 각 3회 = 총 7개 이상)
    if len(state["messages"]) > 6:
        return END
    return "reflect"


# 그래프 구축
builder = StateGraph(MessagesState)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()


if __name__ == "__main__":
    initial_state = {
        "messages": [
            HumanMessage(content="오늘날 '어린 왕자'가 왜 중요한지에 대해 에세이를 작성하세요.")
        ]
    }

    # 그래프 실행
    for output in graph.stream(initial_state):
        node_name = next(iter(output))
        print("\nNew message:", output[node_name]["messages"][-1].content[:100], "...")
