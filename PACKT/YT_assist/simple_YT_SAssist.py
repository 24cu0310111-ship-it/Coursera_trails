import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_api_key() -> None:
	load_dotenv(find_dotenv())
	api_key = os.getenv("SarvamAI_API_KEY")
	if not api_key:
		raise ValueError("SarvamAI_API_KEY not found in .env file")
	os.environ["OPENAI_API_KEY"] = api_key


def build_retriever(video_url: str) -> BM25Retriever:
	transcript = YoutubeLoader.from_youtube_url(video_url).load()
	chunks = RecursiveCharacterTextSplitter(
		chunk_size=1000,
		chunk_overlap=100,
	).split_documents(transcript)

	retriever = BM25Retriever.from_documents(chunks)
	retriever.k = 4
	return retriever


def answer_question(retriever: BM25Retriever, question: str) -> str:
	docs = retriever.invoke(question)
	context = "\n\n".join(doc.page_content for doc in docs)

	llm = ChatOpenAI(
		base_url="https://api.sarvam.ai/v1",
		model="sarvam-m",
		temperature=0.2,
	)

	prompt = f"""
You are a helpful assistant for YouTube videos.
Use only the transcript context below to answer.
If the answer is not in context, reply: I don't know.

Transcript context:
{context}

Question: {question}
"""

	response = llm.invoke(prompt)
	content = response.content if hasattr(response, "content") else response

	if isinstance(content, str):
		return content.strip()

	if isinstance(content, list):
		parts = []
		for item in content:
			if isinstance(item, str):
				parts.append(item)
			elif isinstance(item, dict):
				text = item.get("text")
				if isinstance(text, str):
					parts.append(text)
		return " ".join(parts).strip()

	return str(content)


def main() -> None:
	st.title("Simple YouTube Assistant (Sarvam AI)")
	setup_api_key()

	video_url = st.text_input("YouTube URL")

	if "retriever" not in st.session_state:
		st.session_state.retriever = None

	if st.button("Load Video") and video_url:
		with st.spinner("Loading transcript..."):
			st.session_state.retriever = build_retriever(video_url)
		st.success("Transcript loaded.")

	question = st.text_input("Ask about the video")
	if st.button("Ask") and question:
		if st.session_state.retriever is None:
			st.warning("Load a video first.")
			return

		with st.spinner("Thinking..."):
			answer = answer_question(st.session_state.retriever, question)
		st.write(answer)


if __name__ == "__main__":
	main()
