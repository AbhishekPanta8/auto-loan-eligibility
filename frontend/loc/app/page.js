import Image from "next/image"
// import { LineOfCreditForm } from "@/components/line-of-credit-form"

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <header className="bg-[#3d8b37] text-white py-4">
        <div className="container mx-auto px-4 flex items-center justify-between">
          <div className="flex items-center space-x-8">
            <Image src="/td-logo-en.png" alt="TD Bank Logo" width={60} height={40} className="bg-white p-1 rounded" />
            <nav className="hidden md:flex space-x-6 font-medium">
              <a href="#" className="hover:underline">
                Personal
              </a>
              <a href="#" className="hover:underline">
                Small Business
              </a>
              <a href="#" className="hover:underline">
                Commercial
              </a>
              <a href="#" className="hover:underline">
                Investing
              </a>
              <a href="#" className="hover:underline">
                About TD
              </a>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">Line of Credit Pre-Qualification</h1>
          <p className="text-gray-600 mb-8 text-center">
            Find out if you qualify for a TD Line of Credit and what your potential credit limit and interest rate might
            be.
          </p>

          
        </div>
      </main>
    </div>
  )
}

