import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "公式竞技场 · Formula Arena",
  description: "科学公式发现竞技场 — 赛题发布、选手提交、自动评估、扫地僧解题",
  keywords: ["符号回归", "公式发现", "扫地僧", "竞技场", "SR"],
  authors: [{ name: "Formula Arena" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground min-h-screen`}
      >
        <div className="min-h-screen flex flex-col">
          <header className="border-b bg-card/50 backdrop-blur sticky top-0 z-50">
            <div className="container mx-auto max-w-7xl px-4 h-14 flex items-center justify-between">
              <Link href="/" className="flex items-center gap-2 font-bold">
                <span className="text-xl">⚗️</span>
                <span className="bg-gradient-to-r from-indigo-500 to-purple-500 bg-clip-text text-transparent">
                  公式竞技场
                </span>
                <span className="text-xs text-muted-foreground font-normal hidden sm:inline">
                  Formula Arena
                </span>
              </Link>
              <nav className="flex items-center gap-1 text-sm">
                <Link
                  href="/"
                  className="px-3 py-1.5 rounded-md hover:bg-accent transition-colors"
                >
                  赛题大厅
                </Link>
                <Link
                  href="/publish"
                  className="px-3 py-1.5 rounded-md hover:bg-accent transition-colors"
                >
                  发布赛题
                </Link>
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-1.5 rounded-md hover:bg-accent transition-colors text-muted-foreground"
                >
                  关于
                </a>
              </nav>
            </div>
          </header>
          <main className="flex-1">{children}</main>
          <footer className="border-t py-4 text-center text-xs text-muted-foreground">
            公式竞技场 · 内置扫地僧解题 baseline · 符号回归 + LLM + 深度调研
          </footer>
        </div>
        <Toaster />
      </body>
    </html>
  );
}
