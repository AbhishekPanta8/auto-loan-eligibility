import Image from "next/image"
import { LineOfCreditForm } from "@/components/line-of-credit-form"

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
          <div className="flex items-center space-x-4">
            <select className="bg-[#3d8b37] text-white border border-white rounded px-2 py-1 text-sm">
              <option>English</option>
              <option>Français</option>
            </select>
          </div>
        </div>
      </header>

      <nav className="border-b shadow-sm">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-8">
            <a href="#" className="font-medium text-gray-800 hover:text-[#3d8b37]">
              My Accounts
            </a>
            <a href="#" className="font-medium text-gray-800 hover:text-[#3d8b37]">
              Products
            </a>
            <a href="#" className="font-medium text-gray-800 hover:text-[#3d8b37]">
              Payment Solutions
            </a>
            <a href="#" className="font-medium text-gray-800 hover:text-[#3d8b37]">
              Learn
            </a>
          </div>
          <div className="flex items-center space-x-4">
            <button className="text-gray-600 hover:text-[#3d8b37]">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </button>
            <button className="text-gray-600 hover:text-[#3d8b37]">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </button>
            <button className="flex items-center space-x-1 text-[#3d8b37] font-medium">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                />
              </svg>
              <span>Login</span>
            </button>
          </div>
        </div>
      </nav>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">Line of Credit Pre-Qualification</h1>
          <p className="text-gray-600 mb-8 text-center">
            Find out if you qualify for a TD Line of Credit and what your potential credit limit and interest rate might
            be.
          </p>
          <LineOfCreditForm />
        </div>
      </main>

      <footer className="bg-gray-100 py-8 mt-16">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between">
            <div className="mb-6 md:mb-0">
              <Image src="/td-logo-en.png" alt="TD Bank Logo" width={60} height={40} className="bg-white p-1 rounded" />
              <p className="text-gray-600 mt-2">© {new Date().getFullYear()} TD Bank Group</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <div>
                <h3 className="font-medium text-gray-800 mb-3">About Us</h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Careers
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Contact Us
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Accessibility
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium text-gray-800 mb-3">Legal</h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Privacy
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Security
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Terms of Use
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium text-gray-800 mb-3">Support</h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Help Center
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      FAQs
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Report Fraud
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium text-gray-800 mb-3">Connect</h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Facebook
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      Twitter
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-[#3d8b37]">
                      LinkedIn
                    </a>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

