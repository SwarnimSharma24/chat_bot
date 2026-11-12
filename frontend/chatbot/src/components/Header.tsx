import React from "react";
import { User } from "lucide-react";

export default function Header() {
  return (
    <header className="w-full h-16 px-6 bg-white border-b border-slate-200/80 flex items-center justify-end sticky top-0 z-50">
      <button
        type="button"
        aria-label="User Profile"
        className="w-10 h-10 rounded-full bg-slate-100 hover:bg-slate-200/80 flex items-center justify-center text-slate-700 transition-colors shadow-xs"
      >
        <User className="w-5 h-5" />
      </button>
    </header>
  );
}

