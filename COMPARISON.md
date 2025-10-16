# 📊 Old System vs New System Comparison

## Overview

| Feature | Old System (`run_federated.py`) | New System (`run_blend.py`) |
|---------|--------------------------------|----------------------------|
| **Server role** | Starts with random weights | Has baseline model (pre-trained) |
| **Number of clients** | 2 required | 1 required |
| **Aggregation** | Simple average: (W₀ + W₁)/2 | Blending: (1-α)·W_server + α·W_client |
| **Server contribution** | None (just coordinates) | Active participant with own knowledge |
| **Use case** | Multiple independent data silos | Server + external data source |

## Visual Comparison

### OLD SYSTEM (2 clients, no server knowledge)
```
Round 1:
    Server (random W₀)
       ├──► Client 0 (trains) → W₀'
       └──► Client 1 (trains) → W₀''
    
    Server averages: W₁ = (W₀' + W₀'')/2
    
Round 2:
    Server (W₁)
       ├──► Client 0 (trains) → W₁'
       └──► Client 1 (trains) → W₁''
    
    Server averages: W₂ = (W₁' + W₁'')/2
```

**Key point:** Server has NO initial knowledge, just coordinates averaging.

---

### NEW SYSTEM (1 client, server has baseline)
```
Before Round 1:
    Server trains baseline on 10k images → W_baseline
    
Round 1:
    Server (W_baseline)
       └──► Client (trains on 30k) → W_client₁
    
    Server blends: W₁ = 0.5·W_baseline + 0.5·W_client₁
                       ↑                    ↑
                  server keeps         client's new
                  its knowledge        knowledge
    
Round 2:
    Server (W₁ - already improved!)
       └──► Client (trains on 30k) → W_client₂
    
    Server blends: W₂ = 0.5·W₁ + 0.5·W_client₂
```

**Key point:** Server PARTICIPATES with its own model, doesn't just average.

## Code Comparison

### OLD: Server Initialization
```python
# server_with_save.py
# Server starts with RANDOM weights (no knowledge)
strategy = SaveModelStrategy(
    min_fit_clients=2,  # Need 2 clients
    # No baseline model!
)
```

### NEW: Server Initialization
```python
# server_blend.py
# Server starts with BASELINE MODEL (has knowledge)
baseline_model = create_baseline_model()  # Pre-trained!
initial_parameters = [val.numpy() for val in baseline_model.state_dict().values()]

strategy = BlendServerClient(
    alpha=0.5,
    initial_parameters=initial_parameters,  # Server's knowledge!
    min_fit_clients=1,  # Only need 1 client
)
```

### OLD: Aggregation
```python
# Inside FedAvg (parent class)
def aggregate_fit(self, results):
    # Simple weighted average of all clients
    total_samples = sum(num_samples for _, num_samples in results)
    
    aggregated = []
    for layer in layers:
        weighted_sum = sum(
            weights[layer] * (num_samples / total_samples)
            for weights, num_samples in results
        )
        aggregated.append(weighted_sum)
    
    return aggregated  # Server just passes this through
```

### NEW: Blending
```python
# Inside BlendServerClient
def aggregate_fit(self, results):
    # Get client's weights
    client_weights = super().aggregate_fit(results)
    
    # BLEND with server's current weights
    blended = []
    for w_server, w_client in zip(self._current_parameters, client_weights):
        w_new = (1 - self.alpha) * w_server + self.alpha * w_client
        blended.append(w_new)
    
    self._current_parameters = blended  # Update server's knowledge
    return blended
```

## When to Use Which?

### Use OLD SYSTEM when:
- ✅ You have multiple independent data owners (hospitals, companies, devices)
- ✅ No single party has a good baseline model
- ✅ All parties are equal participants
- ✅ You want pure federated averaging

**Example:** 5 hospitals want to jointly train a model, none has a complete dataset.

---

### Use NEW SYSTEM when:
- ✅ Server already has a decent model (baseline)
- ✅ You want to improve it with external/client data
- ✅ Only one external data source available
- ✅ Server wants to maintain some of its original knowledge

**Example:** You have a pretrained model, customer has private data to improve it.

## Data Distribution

### OLD SYSTEM
```
MNIST Total: 60,000 training images

Client 0: images 0-29,999     (30k images)
Client 1: images 30,000-59,999 (30k images)
Server:   no training data     (0 images)

Result: Learns from all 60k, but split between clients
```

### NEW SYSTEM
```
MNIST Total: 60,000 training images

Server baseline: images 0-9,999    (10k images)
Client:          images 30,000-59,999 (30k images)

Result: Combines 10k (server) + 30k (client) knowledge
        Total: 40k images worth of knowledge
```

## Expected Accuracy Progression

### OLD SYSTEM (2 clients, random start)
```
Round 0: ~10% (random guessing)
Round 1: ~85% (learned from 60k images)
Round 2: ~92%
Round 3: ~95%
Round 5: ~97%
```

### NEW SYSTEM (1 client, baseline start)
```
Round 0: ~88% (baseline already trained)
Round 1: ~93% (incorporated client knowledge)
Round 2: ~95%
Round 3: ~96%
Round 5: ~97%
```

**Key difference:** New system STARTS better because of baseline!

## Which Files to Use?

### Running OLD SYSTEM
```bash
python run_federated.py
# Uses: server_with_save.py + client.py (2 instances)
```

### Running NEW SYSTEM
```bash
python run_blend.py
# Uses: server_blend.py + client_single.py (1 instance)
```

## Hybrid: Best of Both Worlds

You can combine both approaches:

```python
# Server has baseline + multiple clients
class HybridStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, results):
        # Step 1: Average all clients (like OLD system)
        client_avg = super().aggregate_fit(results)
        
        # Step 2: Blend with server baseline (like NEW system)
        blended = []
        for w_server, w_clients in zip(self.baseline, client_avg):
            w_new = (1 - α) * w_server + α * w_clients
            blended.append(w_new)
        
        return blended
```

This works with:
- Server: has baseline model
- Multiple clients: each with private data
- Blending: server maintains some original knowledge

## Summary Table

| Aspect | OLD | NEW |
|--------|-----|-----|
| Clients needed | 2+ | 1+ |
| Server starts with | Random weights | Baseline model |
| Server role | Coordinator | Active participant |
| Formula | W = avg(clients) | W = (1-α)·server + α·client |
| Starting accuracy | ~10% | ~88% |
| Convergence speed | Slower | Faster |
| Server learning | No | Yes (retains knowledge) |
| Best for | Equal partners | Server + helper client |

Choose based on your use case! 🚀


