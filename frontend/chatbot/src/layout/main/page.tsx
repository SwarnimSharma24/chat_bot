"use client";

import React, { useState, useRef, useEffect } from "react";
import { ArrowUp } from "lucide-react";
import { chatApi } from "@/services";
import axios from "axios";

interface SimpleMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export default function MainPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<SimpleMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  console.log(messages, "messages>>>>");

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    const query = input.trim();

    if (!query || isLoading) return;

    setInput("");

    const userMsg: SimpleMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await chatApi(query);

      console.log(response, "response>>>>");

      const answer = response?.answer || "No response received.";

      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: answer,
        },
      ]);
    } catch (error: any) {
      console.error("Chat API Error:", error);

      let errorMessage = "Something went wrong while processing your request.";

      if (error.response) {
        // Backend responded with an HTTP error
        const status = error.response.status;
        const detail = error.response.data?.detail;

        if (status === 503) {
          const backendMessage =
            typeof detail === "string" ? detail : detail?.message;

          if (backendMessage?.toLowerCase().includes("ollama")) {
            errorMessage =
              "AI service is temporarily unavailable. Please try again shortly.";
          } else {
            errorMessage = "Server is busy. Please try again shortly.";
          }
        } else if (status === 401) {
          errorMessage = "Your session has expired. Please log in again.";
        } else if (status === 422) {
          errorMessage = "Invalid request. Please check your message.";
        } else if (status >= 500) {
          errorMessage =
            "Something went wrong on the server. Please try again later.";
        } else {
          errorMessage =
            typeof detail === "string"
              ? detail
              : "Unable to process your request.";
        }
      } else if (error.request) {
        // Request was sent, but no response was received
        errorMessage = "Failed to connect to the backend server.";
      } else {
        // Error occurred while preparing the request
        errorMessage = "Unable to send your message. Please try again.";
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "assistant",
          content: errorMessage,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  };

  const hasMessages = messages.length > 0;

  return (
    <main className="flex-1 flex flex-col justify-between max-w-3xl w-full mx-auto px-4 py-6 min-h-[calc(100vh-4rem)]">
      {/* Messages area (if any) */}
      {hasMessages ? (
        <div className="flex-1 overflow-y-auto space-y-4 py-4 mb-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-slate-900 text-white rounded-br-xs"
                    : "bg-white text-slate-800 border border-slate-200/80 rounded-bl-xs shadow-xs"
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-slate-200/80 px-4 py-2.5 rounded-2xl rounded-bl-xs text-xs text-slate-400 flex items-center gap-1.5 shadow-xs">
                <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center" />
      )}

      {/* Simple Textbar */}
      <div className="w-full pb-4">
        <form
          onSubmit={handleSend}
          className="relative flex items-center w-full bg-white border border-slate-300 rounded-full shadow-xs px-4 py-2.5 focus-within:border-slate-500 focus-within:ring-2 focus-within:ring-slate-200 transition-all"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1 bg-transparent text-sm text-slate-900 placeholder-slate-400 focus:outline-none"
          />

          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            aria-label="Send"
            className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ml-2 ${
              input.trim() && !isLoading
                ? "bg-slate-900 text-white hover:bg-slate-800 active:scale-95"
                : "bg-slate-100 text-slate-300 cursor-not-allowed"
            }`}
          >
            <ArrowUp className="w-4 h-4 stroke-[2.5]" />
          </button>
        </form>
      </div>
    </main>
  );
}
