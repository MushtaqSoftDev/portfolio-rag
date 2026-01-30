import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// CORE IMPORTS - Using the most stable paths for Node 22
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

dotenv.config();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(cors());
app.use(express.json());

let chain = null;
let fallbackContext = "";

// 1. Initialize Knowledge Base
async function initAI() {
  try {
    console.log("🤖 Loading Knowledge Base...");
    
    // Load all .md files manually to be safe
    const dataDir = path.join(__dirname, 'data');
    const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.md'));
    let fullText = "";
    
    files.forEach(file => {
      fullText += fs.readFileSync(path.join(dataDir, file), 'utf8') + "\n\n";
    });
    
    fallbackContext = fullText; // Saved for emergency fallback

    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 800, chunkOverlap: 100 });
    const docs = await splitter.createDocuments([fullText]);

    const embeddings = new GoogleGenerativeAIEmbeddings({ apiKey: process.env.GOOGLE_API_KEY });
    const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      modelName: "gemini-1.5-flash",
      temperature: 0.3,
    });

    chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
    console.log("✅ RAG System Ready!");
  } catch (err) {
    console.error("⚠️ RAG Init Failed, using Fallback Mode:", err.message);
  }
}

app.post('/api/chat', async (req, res) => {
  const { question } = req.body;
  
  try {
    // TRY 1: Professional RAG
    if (chain) {
      const result = await chain.call({ query: question });
      return res.json({ answer: result.text });
    }

    // TRY 2: Fallback Context Injection (If Vector Store fails)
    const model = new ChatGoogleGenerativeAI({ apiKey: process.env.GOOGLE_API_KEY, modelName: "gemini-1.5-flash" });
    const response = await model.invoke([
      ["system", `You are Mushtaq's Assistant. Use this context: ${fallbackContext}`],
      ["human", question]
    ]);
    res.json({ answer: response.content });

  } catch (error) {
    res.status(500).json({ answer: "I'm having trouble connecting. Please email me at mushtaquok70@gmail.com!" });
  }
});

app.listen(10000, () => {
  console.log("Server live on port 10000");
  initAI();
});