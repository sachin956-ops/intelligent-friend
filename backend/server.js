import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";
import pdfParse from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

const app = express();
const PORT = 5000;

app.use(cors({ origin: "http://localhost:3000" }));
app.use(express.json());

// File upload setup
const storage = multer.memoryStorage();
const upload = multer({ storage });

// OpenAI client
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// In-memory storage
const files = {}; // fileId → { text, vectorStore }

// Upload PDF
app.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  try {
    const pdfData = await pdfParse(req.file.buffer);
    const text = pdfData.text;

    // Split into chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunks = await splitter.splitText(text);

    // Create embeddings
    const vectorStore = await MemoryVectorStore.fromTexts(
      chunks,
      chunks.map(() => ({})),
      new OpenAIEmbeddings({ apiKey: process.env.OPENAI_API_KEY })
    );

    const fileId = Date.now();
    files[fileId] = { text, vectorStore };

    res.json({ fileId });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to process PDF" });
  }
});

// Ask Question
app.post("/ask", async (req, res) => {
  const { question, fileId } = req.body;

  if (!files[fileId]) return res.status(400).json({ error: "File not found" });

  try {
    const { vectorStore } = files[fileId];

    // Find relevant chunks
    const results = await vectorStore.similaritySearch(question, 3);
    const context = results.map(r => r.pageContent).join("\n\n");

    // Ask OpenAI
    const completion = await openai.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        { role: "system", content: "You are a helpful assistant that answers questions based on PDF content." },
        { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}` }
      ]
    });

    const answer = completion.choices[0].message.content;
    res.json({ answer });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to get answer" });
  }
});

app.listen(PORT, () =>
  console.log(`✅ Backend running at http://localhost:${PORT}`)
);
