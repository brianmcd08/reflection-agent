from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generation_chain, reflection_chain

load_dotenv()


# state
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# nodes
REFLECTION = "reflect"
GENERATION = "generate"


def generation_node(state: MessageGraph) -> MessageGraph:
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph) -> MessageGraph:
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATION, generation_node)
builder.add_node(REFLECTION, reflection_node)
builder.set_entry_point(GENERATION)


def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) > 6:
        return END
    return REFLECTION


builder.add_conditional_edges(
    GENERATION, should_continue, path_map={END: END, REFLECTION: REFLECTION}
)
builder.add_edge(REFLECTION, GENERATION)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
# print(graph.get_graph().draw_ascii())

if __name__ == "__main__":
    print("Reflection agent")
    inputs = HumanMessage(
        content="""Make this tweet better:"
                    @LangChainAI   

            -newly tool Calling feature is seriously underrated

            After a long wait, it's here-making the implementation of agents across different models

            with the same function calls.

            Made a video covering their newest blog post.           
            """
    )

    response = graph.invoke({"messages": [inputs]})
