"use client"

import Image from "next/image"
import { LineOfCreditForm } from "@/components/line-of-credit-form"
import { useEffect, useState } from "react"
import "./styles.css"

// Define constant for Flinks URL to ensure consistency
const FLINKS_URL = "https://demo.flinks.com/v2/?demo=true&customerName=Demo+Company";

export default function Home() {
  // Use client-side only rendering for the Flinks iframe to avoid hydration issues
  const [isMounted, setIsMounted] = useState(false);
  const [flinksLoginId, setFlinksLoginId] = useState(null);
  const [isVerified, setIsVerified] = useState(false);

  // Load loginId from localStorage on component mount
  useEffect(() => {
    setIsMounted(true);
    
    // Check if loginId exists in localStorage
    const storedLoginId = localStorage.getItem('flinksLoginId');
    if (storedLoginId) {
      setFlinksLoginId(storedLoginId);
      setIsVerified(true);
    }
    
    // Add event listener for Flinks messages
    const handleFlinksMessage = (e) => {
      console.log(e.data);
      
      // Check if the message contains loginId
      if (e.data && e.data.loginId) {
        const loginId = e.data.loginId;
        setFlinksLoginId(loginId);
        setIsVerified(true);
        
        // Store loginId in localStorage
        localStorage.setItem('flinksLoginId', loginId);
      }
    };
    
    window.addEventListener('message', handleFlinksMessage);
    
    // Clean up event listener on unmount
    return () => {
      window.removeEventListener('message', handleFlinksMessage);
    };
  }, []);
  
  // Function to unlink the account
  const handleUnlink = () => {
    // Remove loginId from localStorage
    localStorage.removeItem('flinksLoginId');
    
    // Reset state
    setFlinksLoginId(null);
    setIsVerified(false);
  };

  return (
    <div className="min-h-screen bg-white">
      <header className="bg-[#3d8b37] text-white py-4">
        <div className="container mx-auto px-4 flex items-center justify-between">
          <div className="flex items-center space-x-8">
            <Image
              src="/td-logo-en.png"
              alt="TD Bank Logo"
              width={60}
              height={40}
              className="bg-white p-1 rounded" />
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
            <select
              className="bg-[#3d8b37] text-white border border-white rounded px-2 py-1 text-sm">
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
                stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>
            <button className="text-gray-600 hover:text-[#3d8b37]">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </button>
            <button className="flex items-center space-x-1 text-[#3d8b37] font-medium">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
              <span>Login</span>
            </button>
          </div>
        </div>
      </nav>
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">Line of Credit Pre-Qualification</h1>
          <p className="text-gray-600 mb-8 text-center">
            Find out if you qualify for a TD Line of Credit and what your potential credit limit and interest rate might
            be.
          </p>

          <div className="flex flex-col lg:flex-row gap-8">
            <div className="flex-[3] form-container">
              {isVerified && (
                <div className="bg-green-50 border-l-4 border-green-500 p-4 mb-4 rounded-md">
                  <p className="text-green-700 font-medium">
                    Your identity has been verified. Your data will be automatically verified during application submission.
                  </p>
                </div>
              )}
              <LineOfCreditForm flinksLoginId={flinksLoginId} isVerified={isVerified} />
            </div>
            
            <div className="flex-1 border rounded-lg shadow-md p-4 bg-white identity-verification-container">
              <div className="flex justify-between items-center mb-4 w-full">
                <h2 className="text-2xl font-semibold text-gray-800">
                  {isVerified ? "Identity Verified ✓" : "Verify Your Identity"}
                </h2>
                
                {isVerified && (
                  <button 
                    onClick={handleUnlink}
                    className="text-red-600 hover:text-red-800 text-sm font-medium flex items-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7L7 13m0-6l6 6m4-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Unlink Account
                  </button>
                )}
              </div>
              
              {/* Flinks Connect - Only render on client side to avoid hydration mismatch */}
              <div className="iframe-container">
                {isMounted && !isVerified && (
                  <iframe 
                    className="flinksconnect"
                    height="760"
                    src={FLINKS_URL}>
                  </iframe>
                )}
                
                {isVerified && (
                  <div className="verification-success">
                    <div className="bg-green-100 p-6 rounded-full mb-4">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <p className="text-xl font-medium text-green-700 mb-2">Identity Verified Successfully</p>
                    <p className="text-gray-600 text-center mb-4">
                      Your financial data has been securely retrieved and will be used to enhance your application.
                    </p>
                    <p className="text-sm text-gray-500">
                      Login ID: {flinksLoginId ? flinksLoginId.substring(0, 8) + '...' : ''}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
      <footer className="mt-16">
        <div className="border-t border-gray-200 py-8">
          <div className="container mx-auto px-4">
            <div className="flex flex-col md:flex-row justify-between items-center mb-8">
              <div className="text-center md:text-left mb-4 md:mb-0">
                <p className="text-lg font-medium">
                  Need to talk to us directly?{" "}
                  <a href="#" className="text-[#3d8b37] font-bold">
                    Contact us
                  </a>
                </p>
              </div>
              <div>
                <p className="text-gray-700 font-medium">Follow TD</p>
                <div className="flex space-x-4 mt-2 justify-center md:justify-start">
                  <a
                    href="https://twitter.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 rounded-full border border-gray-300 flex items-center justify-center text-gray-600 hover:text-[#3d8b37] hover:border-[#3d8b37]"
                  >
                    <Image src="/x.png" alt="Twitter" width={20} height={20} />
                  </a>
                  <a
                    href="https://www.facebook.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 rounded-full border border-gray-300 flex items-center justify-center text-gray-600 hover:text-[#3d8b37] hover:border-[#3d8b37]"
                  >
                    <Image src="/facebook.png" alt="Facebook" width={20} height={20} />
                  </a>
                  <a
                    href="https://www.instagram.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 rounded-full border border-gray-300 flex items-center justify-center text-gray-600 hover:text-[#3d8b37] hover:border-[#3d8b37]"
                  >
                    <Image src="/instagram.png" alt="Instagram" width={20} height={20} />
                  </a>
                  <a
                    href="https://www.youtube.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 rounded-full border border-gray-300 flex items-center justify-center text-gray-600 hover:text-[#3d8b37] hover:border-[#3d8b37]"
                  >
                    <Image src="/youtube.png" alt="YouTube" width={20} height={20} />
                  </a>
                  <a
                    href="https://www.linkedin.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="w-10 h-10 rounded-full border border-gray-300 flex items-center justify-center text-gray-600 hover:text-[#3d8b37] hover:border-[#3d8b37]"
                  >
                    <Image src="/linkedin.png" alt="LinkedIn" width={20} height={20} />
                  </a>
                </div>

              </div>
            </div>

            <div
              className="grid grid-cols-2 md:grid-cols-7 gap-4 text-sm border-t border-gray-200 pt-6">
              <a href="#" className="text-gray-600 hover:text-[#3d8b37]">
                Privacy & Security
              </a>
              <a href="#" className="text-gray-600 hover:text-[#3d8b37]">
                Legal
              </a>
              <a href="#" className="text-gray-600 hover:text-[#3d8b37]">
                Accessibility
              </a>
              <a href="#" className="text-gray-600 hover:text-[#3d8b37]">
                CDIC Member
              </a>
              <a href="#" className="text-gray-600 hover:text-[#3d8b37]">
                About Us
              </a>
              <a href="#" className="text-gray-600 hover:text-[#3d8b37]">
                Careers
              </a>
              <a href="#" className="text-gray-600 hover:text-[#3d8b37]">
                Manage online experience
              </a>
            </div>

            <div
              className="border-t border-gray-200 mt-6 pt-6 flex flex-col md:flex-row justify-between items-center">
              <p className="text-sm text-gray-600 mb-4 md:mb-0">
                TD Bank Tower - Corporate Office 66 Wellington Street West, Toronto, ON M5K 1A2
              </p>
              <button
                className="flex items-center text-[#3d8b37] border border-[#3d8b37] rounded-full px-4 py-1 text-sm font-medium">
                <Image
                  src="/arrow-up-icon.png"
                  alt="Arrow Up"
                  width={16}
                  height={16}
                  className="mr-1" />
                Top
              </button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

