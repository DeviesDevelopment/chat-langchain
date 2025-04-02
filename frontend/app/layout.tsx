import "./globals.css";
import type { Metadata } from "next";
import { Fira_Sans } from "next/font/google";
import { Providers } from './providers'

const firaSans = Fira_Sans({ 
  subsets: ['latin'], 
  weight: ['400', '700'],
  variable: '--font-fira-sans',
  display: 'swap', });

export const metadata: Metadata = {
  title: "Devies GPT",
  description: "Chatbot for Devies",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${firaSans.className} h-full`}>
      <body className={`${firaSans.className} h-full`}>
        <Providers>
          <div
            className="flex flex-col h-full md:p-8"
            style={{ background: "rgb(38, 38, 41)" }}
          >
            {children}
          </div>
        </Providers>
      </body>
    </html>
  );
}
