# OpenTelemetry Quick Start Guide for Exgentic

## 🚀 Quick Setup (5 minutes)

### 1. Start Jaeger

**Important:** Exgentic automatically checks if the OTEL collector is reachable before starting. If the health check fails, tracing will be disabled with a warning message.

```bash
# Using Podman (or replace with 'docker' if you have Docker)
podman run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

```bash

exgentic setup --agent tool_calling 

exgentic setup --benchmark tau2

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 # typical for self-hosted Jaeger server
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf  # or 'grpc'
export EXGENTIC_OTEL_ENABLED=true


# Optional: Enable content recording (includes task descriptions, observations, etc.)
# Warning: This may include sensitive data in traces
export OTEL_RECORD_CONTENT="true"


exgentic evaluate \
  --benchmark tau2 \
  --agent tool_calling \
  --model openai/Azure/gpt-4o2 \
  --set benchmark.user_simulator_model="openai/Azure/gpt-4o" \
  --task 1 \
  --max-steps 10
```

 

### 3. View Traces
Open browser: **http://localhost:16686**

### 4. Export Traces as JSON

#### Via UI:
1. Go to http://localhost:16686
2. Select service: `exgentic`
3. Click "Find Traces"
4. Click on a trace
5. Click "JSON" button in top-right

#### Via API:
```bash
# Get all recent traces
curl "http://localhost:16686/api/traces?service=exgentic&limit=100" | jq '.' > traces.json

# Get specific trace by ID
curl "http://localhost:16686/api/traces/TRACE_ID" | jq '.' > trace.json
```

## 📊 What You'll See

### Trace Structure
```
Session Span (root)
├── execute_tool: initial_observation
├── execute_tool: action_1
├── execute_tool: action_2
└── execute_tool: action_n
```

### Key Attributes
- `gen_ai.conversation.id` - Session identifier
- `gen_ai.tool.name` - Action/tool name
- `gen_ai.tool.parameters` - Tool input (if OTEL_RECORD_CONTENT=true)
- `gen_ai.tool.result` - Tool output (if OTEL_RECORD_CONTENT=true)
- `exgentic.score.success` - Task success status
- `exgentic.session.steps` - Number of steps taken

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SERVICE_NAME` | `exgentic` | Service name in Jaeger |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | - | Jaeger endpoint (e.g., http://localhost:4318) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Protocol: `http/protobuf` or `grpc` |
| `OTEL_RECORD_CONTENT` | `false` | Include task/observation content in traces |
| `OTEL_SERVICE_VERSION` | `1.0.0` | Service version |
| `DEPLOYMENT_ENVIRONMENT` | `dev` | Environment name |

### Ports

| Port | Service |
|------|---------|
| 16686 | Jaeger UI |
| 4317 | OTLP gRPC |
| 4318 | OTLP HTTP |

## 🛠️ Troubleshooting

### Health Check Warnings

If you see a warning like:
```
WARNING: OTEL collector health check failed: Cannot connect to OTEL collector at localhost:4318.
OpenTelemetry tracing will be DISABLED.
```

This means:
1. Jaeger/OTEL collector is not running or not accessible
2. Tracing has been automatically disabled to prevent errors
3. Your benchmark will run normally, just without trace collection

**To fix:**
- Start Jaeger: `podman run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one:latest`
- Verify it's running: `podman ps | grep jaeger`
- Check connectivity: `curl http://localhost:4318`

### Jaeger not starting?
```bash
# Check if already running
podman ps | grep jaeger

# Stop and remove existing container
podman stop jaeger && podman rm jaeger

# Start fresh
podman run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one:latest
```

### No traces appearing?
1. Check environment variables are set: `env | grep OTEL`
2. Verify Jaeger is running: `podman ps | grep jaeger`
3. Check Jaeger logs: `podman logs jaeger`
4. Ensure benchmark completed successfully

### Traces incomplete?
- Wait a few seconds after benchmark completes
- Check session logs for OTEL errors
- Verify network connectivity to Jaeger

## 📚 Additional Documentation

- **Full Export Guide**: See [`OTEL_EXPORT_INSTRUCTIONS.md`](./OTEL_EXPORT_INSTRUCTIONS.md)
- **Semantic Conventions**: See [`OTEL_SEMANTIC_CONVENTIONS.md`](./OTEL_SEMANTIC_CONVENTIONS.md)
- **Jaeger Docs**: https://www.jaegertracing.io/docs/

## 🧹 Cleanup

```bash
# Stop Jaeger
podman stop jaeger

# Remove Jaeger container
podman rm jaeger

# Remove Jaeger image (optional)
podman rmi jaegertracing/all-in-one:latest