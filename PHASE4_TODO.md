# Phase 4: Disk-Backed Storage - TODO

**Status:** ⏳ Next Phase

## Goal

Enable true 256k context by storing checkpoints on disk (S3/Ceph) instead of RAM.

**Current Limitation:**
- 100 checkpoints × 5MB = 500MB RAM (manageable)
- 256k context / 500 tokens per checkpoint = 512 checkpoints × 5MB = **2.5GB RAM** (too much)

**Solution:**
- Store checkpoints on disk (S3/Ceph)
- RAM only holds current window (~50MB)
- LRU eviction swaps old checkpoints to disk

## Implementation

### Step 1: Disk I/O Layer

```cpp
// Abstract storage interface
struct checkpoint_storage {
    virtual void save(const std::string& key, const std::vector<uint8_t>& data) = 0;
    virtual bool load(const std::string& key, std::vector<uint8_t>& data) = 0;
    virtual void evict(const std::string& key) = 0;
};

// RAM storage (current)
struct ram_storage : checkpoint_storage {
    std::unordered_map<std::string, std::vector<uint8_t>> cache;
};

// S3/Ceph storage
struct s3_storage : checkpoint_storage {
    std::string bucket;
    std::string endpoint;
    // S3 client implementation...
};
```

### Step 2: LRU Eviction

```cpp
struct checkpoint_entry {
    std::string key;
    std::string semantic_name;
    int32_t pos_min, pos_max;
    bool in_ram;
    uint64_t last_accessed;
};

// When RAM full:
void evict_to_disk() {
    // Find LRU checkpoint not in use
    auto* entry = find_lru_entry();
    if (entry && entry->in_ram) {
        s3_storage->save(entry->key, entry->data);
        entry->in_ram = false;
        free_ram(entry->data.size());
    }
}
```

### Step 3: Integration

```cpp
// In apply_checkpoint():
auto* cp = find_checkpoint_by_name(slot, target_name);
if (cp && !cp->in_ram) {
    // Load from disk
    s3_storage->load(cp->key, cp->data);
    cp->in_ram = true;
    evict_to_disk();  // Make room
}
restore_checkpoint(*cp);
```

## Files to Create

- `examples/server/server-storage.h/cpp` - Storage abstraction
- `examples/server/server-s3-storage.cpp` - S3 implementation (optional, use cURL)
- `examples/server/server-disk-storage.cpp` - Local disk fallback

## CLI Flags

```cpp
options.push_back({ "*", "--checkpoint-storage TYPE", "Storage type: ram, disk, s3 (default: ram)" });
options.push_back({ "*", "--checkpoint-path PATH", "Path for disk storage (default: ~/.llama/checkpoints/)" });
options.push_back({ "*", "--checkpoint-bucket NAME", "S3 bucket name for s3 storage" });
options.push_back({ "*", "--checkpoint-max-ram MB", "Max RAM for checkpoints (default: 512)" });
```

## Testing

After completion:
- 256k context with 500 checkpoints (100 in RAM, 400 on disk)
- RAM usage: ~500MB (constant, regardless of context size)
- Disk usage: ~2.5GB for full 256k context

*tail flicks* Phase 4 will unlock true 256k agentic contexts!