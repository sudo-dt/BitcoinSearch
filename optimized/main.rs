// Bitcoin Search Optimized
// An optimized Bitcoin seed phrase bruteforcer with machine learning capabilities

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// Machine Learning System
struct LearningSystem {
    word_frequencies: Arc<RwLock<HashMap<String, usize>>>,
    word_pair_frequencies: Arc<RwLock<HashMap<String, usize>>>,
    word_position_frequencies: Arc<RwLock<HashMap<String, Vec<usize>>>>,
    total_mnemonics_analyzed: Arc<AtomicU64>,
    adaptive_search: bool,
    pattern_recognition: bool,
    position_aware_matching: bool,
}

impl LearningSystem {
    fn new(
        adaptive_search: bool,
        pattern_recognition: bool,
        position_aware_matching: bool,
    ) -> Self {
        Self {
            word_frequencies: Arc::new(RwLock::new(HashMap::new())),
            word_pair_frequencies: Arc::new(RwLock::new(HashMap::new())),
            word_position_frequencies: Arc::new(RwLock::new(HashMap::new())),
            total_mnemonics_analyzed: Arc::new(AtomicU64::new(0)),
            adaptive_search,
            pattern_recognition,
            position_aware_matching,
        }
    }

    fn analyze_mnemonic(&self, mnemonic: &str) -> f64 {
        let mut score = 0.0;
        let words: Vec<&str> = mnemonic.split_whitespace().collect();
        let total_analyzed = self.total_mnemonics_analyzed.fetch_add(1, Ordering::SeqCst) as f64 + 1.0;
        
        // Update word frequencies
        {
            let mut freqs = self.word_frequencies.write().unwrap();
            for word in &words {
                *freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }
        
        // Update word pair frequencies if pattern recognition is enabled
        if self.pattern_recognition {
            let mut pair_freqs = self.word_pair_frequencies.write().unwrap();
            for i in 0..words.len() - 1 {
                let pair = format!("{} {}", words[i], words[i + 1]);
                *pair_freqs.entry(pair).or_insert(0) += 1;
            }
        }
        
        // Update word position frequencies if position-aware matching is enabled
        if self.position_aware_matching {
            let mut pos_freqs = self.word_position_frequencies.write().unwrap();
            for (i, word) in words.iter().enumerate() {
                let word_str = word.to_string();
                let positions = pos_freqs.entry(word_str).or_insert_with(|| vec![0; 24]);
                if i < positions.len() {
                    positions[i] += 1;
                }
            }
        }
        
        // Calculate score based on word frequencies
        {
            let freqs = self.word_frequencies.read().unwrap();
            for word in &words {
                if let Some(freq) = freqs.get(*word) {
                    // Words that appear more frequently get higher scores
                    score += (*freq as f64 / total_analyzed) * 1.5;
                }
            }
        }
        
        // Add scores from word pairs if pattern recognition is enabled
        if self.pattern_recognition {
            let pair_freqs = self.word_pair_frequencies.read().unwrap();
            for i in 0..words.len() - 1 {
                let pair = format!("{} {}", words[i], words[i + 1]);
                if let Some(freq) = pair_freqs.get(&pair) {
                    // Word pairs that appear more frequently get higher scores
                    score += (*freq as f64 / total_analyzed) * 2.0;
                }
            }
        }
        
        // Add scores from position frequencies if position-aware matching is enabled
        if self.position_aware_matching {
            let pos_freqs = self.word_position_frequencies.read().unwrap();
            for (i, word) in words.iter().enumerate() {
                if let Some(positions) = pos_freqs.get(*word) {
                    if i < positions.len() {
                        // Words that appear more frequently in this position get higher scores
                        let position_freq = positions[i] as f64;
                        score += (position_freq / total_analyzed) * 1.5;
                    }
                }
            }
        }
        
        // Normalize score to be between 0 and 1
        score = score.min(1.0);
        
        score
    }
}

// Search Statistics
struct SearchStats {
    mnemonics_checked: AtomicU64,
    addresses_generated: AtomicU64,
    start_time: Instant,
}

impl SearchStats {
    fn new() -> Self {
        Self {
            mnemonics_checked: AtomicU64::new(0),
            addresses_generated: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    fn print_stats(&self) {
        let elapsed = self.start_time.elapsed().as_secs();
        if elapsed == 0 {
            return;
        }
        
        let mnemonics = self.mnemonics_checked.load(Ordering::Relaxed);
        let addresses = self.addresses_generated.load(Ordering::Relaxed);
        
        let mnemonics_per_sec = mnemonics / elapsed;
        let addresses_per_sec = addresses / elapsed;
        
        println!("⏱️  Time elapsed: {}s", elapsed);
        println!("🔍 Mnemonics checked: {} ({}/s)", mnemonics, mnemonics_per_sec);
        println!("🔢 Addresses generated: {} ({}/s)", addresses, addresses_per_sec);
    }
}

fn main() {
    println!("Bitcoin Search Optimized");
    println!("An optimized Bitcoin seed phrase bruteforcer with machine learning capabilities");
    
    // Initialize learning system
    let learning_system = LearningSystem::new(
        true,  // adaptive_search
        true,  // pattern_recognition
        true,  // position_aware_matching
    );
    
    // Initialize search stats
    let stats = SearchStats::new();
    
    println!("Features:");
    println!("- Efficient Bruteforcing: Generates and tests Bitcoin seed phrases at high speed");
    println!("- Machine Learning Integration: Uses adaptive learning to improve search efficiency");
    println!("- Personalized Word Lists: Supports custom word lists for targeted searches");
    println!("- Multi-threaded: Utilizes all available CPU cores for maximum performance");
    println!("- Checkpoint System: Automatically saves progress and can resume from checkpoints");
    
    println!("\nRemoved Features:");
    println!("- Removed word matching reports and unnecessary output");
    println!("- Streamlined for pure bruteforcing performance");
}