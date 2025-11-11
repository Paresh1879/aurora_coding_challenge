#!/bin/bash

# Simple test script for the Member Q&A API

BASE_URL="${1:-https://aurora-coding-challenge.onrender.com/}"

echo "Testing Member Q&A API at $BASE_URL"
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# Test root endpoint
echo "2. Testing root endpoint..."
curl -s "$BASE_URL/" | python3 -m json.tool
echo ""

# Test question 1
echo "3. Testing question: 'Wheat is Sophia planning to do in the Amalfi Coast?'"
curl -s -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Wheat is Sophia planning to do in the Amalfi Coast?"}' | python3 -m json.tool
echo ""

# Test question 2
echo "4. Testing question: 'How many cars does Vikram Desai have??'"
curl -s -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many cars does Vikram Desai have?"}' | python3 -m json.tool
echo ""

# Test question 3
echo "5. Testing question: 'What kind of Decor does Fatima like?'"
curl -s -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What kind of Decor does Fatima like?"}' | python3 -m json.tool
echo ""

echo "Testing complete!"

