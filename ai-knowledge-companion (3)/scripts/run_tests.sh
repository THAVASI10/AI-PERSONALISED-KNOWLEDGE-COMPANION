#!/bin/bash

# Comprehensive test runner for AI Knowledge Companion
echo "Starting AI Knowledge Companion Test Suite..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Setup environment if needed
echo "Setting up environment..."
python3 scripts/setup_environment.py

if [ $? -ne 0 ]; then
    echo "Environment setup failed!"
    exit 1
fi

# Start the backend server in background
echo "Starting FastAPI backend..."
python3 main.py &
BACKEND_PID=$!

# Wait a moment for server to start
sleep 5

# Run system tests
echo "Running system tests..."
python3 scripts/test_system.py
TEST_RESULT=$?

# Stop the backend server
echo "Stopping backend server..."
kill $BACKEND_PID 2>/dev/null

# Report results
if [ $TEST_RESULT -eq 0 ]; then
    echo "All tests passed! System is ready for use."
    exit 0
else
    echo "Some tests failed. Check logs for details."
    exit 1
fi
