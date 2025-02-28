import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LineOfCreditForm } from "./line-of-credit-form";
import "@testing-library/jest-dom";

// Mock the fetch function
global.fetch = jest.fn(() =>
  Promise.resolve({
    json: () => Promise.resolve({ approved: 1, approved_amount: 25000, interest_rate: 5.5 }),
  })
);



describe("LineOfCreditForm", () => {
  beforeEach(() => {
    render(<LineOfCreditForm />);
  });

  test("renders the initial form", () => {
    expect(screen.getByText("Line of Credit Application")).toBeInTheDocument();
    expect(screen.getByText("Documents You'll Need")).toBeInTheDocument();
    expect(screen.getByLabelText("Full Name")).toBeInTheDocument();
    expect(screen.getByLabelText("Your Age")).toBeInTheDocument();
    expect(screen.getByLabelText("Province")).toBeInTheDocument();
  });

  test("navigates through form steps", async () => {
    // Step 1 to Step 2
    fireEvent.click(screen.getByText("Continue"));
    await waitFor(() => {
      expect(screen.getByText("Employment & Income Information")).toBeInTheDocument();
    });

    // Step 2 to Step 3
    fireEvent.click(screen.getByText("Continue"));
    await waitFor(() => {
      expect(screen.getByText("Financial Information")).toBeInTheDocument();
    });

    // Step 3 to Step 4
    fireEvent.click(screen.getByText("Continue"));
    await waitFor(() => {
      expect(screen.getByText("Credit Information")).toBeInTheDocument();
    });

    // Step 4 to Step 5
    fireEvent.click(screen.getByText("Continue"));
    await waitFor(() => {
      expect(screen.getByText("Line of Credit Request")).toBeInTheDocument();
    });

    // Step 5 to Step 6 (submission)
    fireEvent.click(screen.getByText("Submit Application"));
    await waitFor(() => {
      expect(screen.getByText("Congratulations! Your application is pre-approved")).toBeInTheDocument();
    });
  });

  test("handles form input changes", () => {
    const fullNameInput = screen.getByLabelText("Full Name");
    fireEvent.change(fullNameInput, { target: { value: "John Doe" } });
    expect(fullNameInput).toHaveValue("John Doe");

    const ageInput = screen.getByLabelText("Your Age");
    fireEvent.change(ageInput, { target: { value: "30" } });
    expect(ageInput).toHaveValue(30);
  });

  test("displays tooltips on hover", async () => {

    const tooltipTriggers = screen.getAllByTestId("tooltip-full-name"); // Get all matching elements
    const tooltipTrigger = tooltipTriggers[0]; // Use the first tooltip
    //console.log(screen.debug());

    fireEvent.focus(tooltipTrigger); // Try focus instead of hover
    fireEvent.keyDown(tooltipTrigger, { key: "Enter", code: "Enter" });

    fireEvent.mouseEnter(tooltipTrigger);
    await waitFor(() => {
      expect(screen.getAllByText(/Enter your full legal name/)[0]).toBeInTheDocument();
    });
  });  
  

  test("handles select inputs", async () => {
    const provinceSelect = screen.getByLabelText("Province");
    fireEvent.mouseDown(provinceSelect);
    const ontarioOption = await screen.findByText("Ontario");
    fireEvent.click(ontarioOption);
    expect(provinceSelect).toHaveTextContent("Ontario");
  });

  test("submits form and displays result", async () => {
    // Navigate through all steps
    for (let i = 0; i < 4; i++) {
      fireEvent.click(screen.getByText("Continue"));
    }

    // Submit the form
    fireEvent.click(screen.getByText("Submit Application"));

    await waitFor(() => {
      expect(screen.getByText("Congratulations! Your application is pre-approved")).toBeInTheDocument();
      expect(screen.getByText("$25,000")).toBeInTheDocument(); // Approved amount
      expect(screen.getByText("5.50%")).toBeInTheDocument(); // Interest rate
    });
  });

  test("handles form submission error", async () => {
    // Mock a fetch error
    global.fetch = jest.fn(() => Promise.reject("API error"));

    // Navigate through all steps
    for (let i = 0; i < 4; i++) {
      fireEvent.click(screen.getByText("Continue"));
    }

    // Submit the form
    fireEvent.click(screen.getByText("Submit Application"));

    await waitFor(() => {
      expect(screen.queryByText("Congratulations! Your application is pre-approved")).not.toBeInTheDocument();
    });
  });
});
