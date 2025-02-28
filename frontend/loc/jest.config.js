const nextJest = require("next/jest");

const createJestConfig = nextJest({
  dir: "./",
});

const customJestConfig = {
  setupFilesAfterEnv: ["<rootDir>/jest.setup.js"], // Setup file for Jest
  testEnvironment: "jest-environment-jsdom", // Set environment to JSDOM
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1", // Alias for Next.js path resolution
  },
  transform: {
    "^.+\\.(js|jsx|ts|tsx)$": ["babel-jest", { presets: ["next/babel"] }],
  },
};

module.exports = createJestConfig(customJestConfig);
