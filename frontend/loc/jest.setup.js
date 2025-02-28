import "@testing-library/jest-dom"; // Extends Jest matchers
import "whatwg-fetch"; // Fixes fetch() issues in tests

// Mock ResizeObserver for Radix UI
global.ResizeObserver = class {
    constructor() {}
    observe() {}
    unobserve() {}
    disconnect() {}
};

global.PointerEvent = class PointerEvent extends Event {};
