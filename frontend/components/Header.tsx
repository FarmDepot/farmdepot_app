"use client";

import Link from "next/link";
import Image from "next/image";

export default function Header() {
  return (
    <header className="w-full bg-white shadow-md sticky top-0 z-50">
      <div className="max-w-6xl mx-auto flex items-center justify-between p-4">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-2">
          <Image
            src="/farmdepot-logo.png"
            alt="FarmDepot Logo"
            width={40}
            height={40}
          />
          <span className="text-xl font-bold text-green-700">FarmDepot</span>
        </Link>

        {/* Navigation */}
        <nav className="flex space-x-6">
          <Link
            href="/"
            className="text-gray-700 hover:text-green-700 font-medium"
          >
            Home
          </Link>
          <Link
            href="/list"
            className="text-gray-700 hover:text-green-700 font-medium"
          >
            Add Product
          </Link>
        </nav>
      </div>
    </header>
  );
}