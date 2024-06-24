// import { GithubRepoLoader } from "langchain/document_loaders/web/github";
// // Peer dependency, used to support .gitignore syntax
// import ignore from "ignore";

// // Will not include anything under "ignorePaths"
// const loader = new GithubRepoLoader(
//     "https://github.com/langchain-ai/langchainjs",
//     { recursive: false, ignorePaths: ["*.md", "yarn.lock"] }
//   );

// const docs = await loader.load();

// console.log(docs.slice(0, 3));










// import { GithubRepoLoader } from "langchain/document_loaders/web/github";
// import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

// async function loadDocuments() {
//   // Load documents from the GitHub repository
//   const githubLoader = new GithubRepoLoader(
//     "https://github.com/langchain-ai/langchainjs",
//     { recursive: false, ignorePaths: ["*.md", "yarn.lock"] }
//   );

//   const githubDocs = await githubLoader.load();
//   console.log("GitHub Docs:", githubDocs.slice(0, 3));

//   // Load documents from the PDF file
//   const pdfLoader = new PDFLoader("./files/MachineLearning-Lecture01.pdf");
//   const pdfDocs = await pdfLoader.load();
//   console.log("PDF Docs:", pdfDocs.slice(0, 5));

//   // Combine documents from both sources
//   const allDocs = [...githubDocs, ...pdfDocs];
//   return allDocs;
// }

// (async () => {
//   const documents = await loadDocuments();
//   console.log("Combined Docs:", documents.slice(0, 5));

//   // You can now process `documents` as needed
// })();




import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { 
    RecursiveCharacterTextSplitter
} from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";

import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();


if (!process.env.OPENAI_API_KEY) {
  throw new Error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.");
}

const embeddings = new OpenAIEmbeddings({
  configuration: {
    baseURL: "https://tlx-ai-hackathon-2024.azure-api.net/gpt-35-turbo/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-15-preview HTTP/1.1",
  },
  apiKey: process.env.OPENAI_API_KEY
});


const loader = new PDFLoader("./files/MachineLearning-Lecture01.pdf");

const rawCS229Docs = await loader.load();
// console.log(rawCS229Docs.slice(0, 5));


const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 128,
    chunkOverlap: 0,
  });

  const splitDocs = await splitter.splitDocuments(rawCS229Docs);

  const vectorstore = new MemoryVectorStore(embeddings);
  await vectorstore.addDocuments(splitDocs);

 
  const retriever = vectorstore.asRetriever();


 

const convertDocsToString = (documents) => {
  return documents.map((document) => {
    return `<doc>\n${document.pageContent}\n</doc>`
  }).join("\n");
};

/*
{
question: "What is deep learning?"
}
*/

const documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
]);


const results = await documentRetrievalChain.invoke({
  question: "What are the prerequisites for this course?"
});
console.log(results);