# feat: add comprehensive health check endpoint

## Summary
- GET /health with dependency status checks
- Database connectivity verification
- Cache availability check
- Upstream service health aggregation

## Test plan
- [x] Health check returns 200 when all deps healthy
- [x] Returns 503 when critical dep is down
- [x] Response time < 100ms unde
