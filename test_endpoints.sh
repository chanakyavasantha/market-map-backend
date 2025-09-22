#!/bin/bash

# MarketMap API Endpoint Testing Script
# Make sure your backend server is running on localhost:8000

#BASE_URL="http://localhost:8000"
BASE_URL="https://market-map-backend-hcezehhmakf8exca.canadacentral-01.azurewebsites.net/"
echo "Testing MarketMap API endpoints..."
echo "Base URL: $BASE_URL"
echo "==========================================="

# Test 1: Root endpoint
echo "\n1. Testing root endpoint..."
curl -s -w "\nStatus: %{http_code}\n" "$BASE_URL/"

# Test 2: Debug environment endpoint
echo "\n2. Testing debug environment..."
curl -s -w "\nStatus: %{http_code}\n" "$BASE_URL/debug/env"

# Test 3: Indices status endpoint (the one that was failing)
echo "\n3. Testing indices status endpoint..."
curl -s -w "\nStatus: %{http_code}\n" "$BASE_URL/api/indices/status"

# Test 4: Last updated endpoint
echo "\n4. Testing last updated endpoint..."
curl -s -w "\nStatus: %{http_code}\n" "$BASE_URL/api/indices/last-updated"

# Test 5: All indices endpoint
echo "\n5. Testing all indices endpoint..."
curl -s -w "\nStatus: %{http_code}\n" "$BASE_URL/api/indices"

# Test 6: Specific index endpoints (common market indices)
echo "\n6. Testing specific index endpoints..."
INDICES=("^GSPC" "^DJI" "^IXIC" "^FTSE" "^N225")

for index in "${INDICES[@]}"; do
    echo "\n   Testing $index..."
    curl -s -w "\nStatus: %{http_code}\n" "$BASE_URL/api/indices/$index"
done

# Test 7: Health check with detailed output
echo "\n7. Detailed health check..."
echo "Testing with verbose output:"
curl -v "$BASE_URL/api/indices/status" 2>&1 | head -20

echo "\n==========================================="
echo "Testing complete!"
echo "\nIf you see 500 errors, check:"
echo "1. MongoDB connection (MONGODB_URI environment variable)"
echo "2. Backend server logs for detailed error messages"
echo "3. SSL/TLS configuration in dependencies.py"