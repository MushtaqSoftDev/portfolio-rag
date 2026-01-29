import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/core/vectorstores";
import { RetrievalQAChain } from "langchain/chains";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

let chain;

// Initialize the RAG system
async function initRAG() {
  try {
    console.log("--- Initializing RAG Knowledge Base ---");
    
    // 1. Load all .md files from the data folder
    const loader = new DirectoryLoader(
      path.join(__dirname, 'data'),
      { ".md": (path) => new TextLoader(path) }
    );
    const docs = await loader.load();

    // 2. Split text into chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splitDocs = await splitter.splitDocuments(docs);

    // 3. Create Embeddings & Vector Store
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
    });
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

    // 4. Setup LLM and Chain
    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      modelName: "gemini-1.5-flash",
      temperature: 0.3,
    });

    chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
    console.log("--- RAG Ready: Knowledge Base Indexed ---");
  } catch (error) {
    console.error("Failed to initialize RAG:", error);
  }
}

app.post('/api/chat', async (req, res) => {
  if (!chain) return res.status(503).json({ error: "System initializing, please wait." });
  
  try {
    const { question } = req.body;
    const result = await chain.call({ query: question });
    res.json({ answer: result.text });
  } catch (error) {
    res.status(500).json({ error: "AI failed to respond." });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  initRAG(); // Run indexer on startup
});