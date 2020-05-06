use std::collections::HashMap;
use std::convert::From;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::{
    hash::Hash,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, TryLockError,
    },
    time::Duration,
};
use log::trace;

#[derive(Debug)]
pub enum CacheError {
    AccessError,
    GetError,
    ListError,
}

impl Display for CacheError {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self)
    }
}

impl<T> From<TryLockError<T>> for CacheError {
    fn from(_: TryLockError<T>) -> Self {
        CacheError::AccessError
    }
}

#[derive(Debug)]
pub struct CacheEntry<T> {
    data: T,
    date: chrono::DateTime<chrono::Utc>,
}

impl<T> CacheEntry<T> {
    pub fn new(data: T) -> Self {
        CacheEntry {
            data,
            date: chrono::Utc::now(),
        }
    }
}

/// A concurrently safe Cache with a ttl cleaning thread.
/// 
/// Values are removed after a while of beeing unused (reset by the `add` function).
pub struct ConcurrentTTLCache<K, V> {
    data: Arc<Mutex<HashMap<K, CacheEntry<V>>>>,
    running: Arc<AtomicBool>,
}

impl<K, V> ConcurrentTTLCache<K, V>
where
    K: Eq + Hash + Send + 'static,
    V: Clone + Debug + Send + 'static,
{
    /// Creates a new instance of the cache, spawning a thread to handle the cleaning.
    /// `ttl_ms` is the max duration (in millis) that an item might be in the cache
    pub fn new(ttl_ms: i32) -> Self {
        let data = Arc::new(Mutex::new(HashMap::new()));
        let running = Arc::new(AtomicBool::new(true));

        ConcurrentTTLCache::clean(data.clone(), running.clone(), ttl_ms);

        ConcurrentTTLCache { data, running }
    }

    /// Spawns a thread to delete all entries that having been used in the last ttl_ms millis. 
    fn clean(hs: Arc<Mutex<HashMap<K, CacheEntry<V>>>>, running: Arc<AtomicBool>, ttl_ms: i32) {
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                let now = chrono::Utc::now();
                let mut lock = hs.lock().unwrap();

                lock.retain(|_, entry| {
                    now.signed_duration_since(entry.date)
                        < chrono::Duration::milliseconds(ttl_ms as i64)
                });

                // drop would only be dropped when thread terminates
                // which would be entirity of the lifetime of the cache.
                drop(lock);
                std::thread::sleep(Duration::from_millis(ttl_ms as u64))
            }
            trace!("Cleaning thread has stopped!");
        });
    }

    /// Inserts a new entry into the cache
    pub fn put(&mut self, key: K, data: V) -> Result<(), CacheError> {
        let entry = CacheEntry::new(data);

        self.data.try_lock()?.insert(key, entry);
        Ok(())
    }

    /// Get's a value from the cache. If none is found returns a ```CacheError::GetError```
    pub fn get(&self, key: &K) -> Result<V, CacheError> {
        match self.data.try_lock()?.get_mut(key) {
            Some(e) => {
                e.date = chrono::Utc::now();
                Ok(e.data.clone())
            }
            None => Err(CacheError::GetError),
        }
    }

    /// Removes an entry from the cache
    pub fn delete(&mut self, key: &K) -> Result<(), CacheError> {
        self.data.try_lock()?.remove(key);
        Ok(())
    }

    /// Clears the entire cache data
    pub fn clear(&mut self) -> Result<(), CacheError> {
        self.data.try_lock()?.clear();

        Ok(())
    }

    /// Stops the cleaning thread
    pub fn stop_clear(&mut self) -> Result<(), CacheError> {
        self.running.swap(false, Ordering::Relaxed);
        Ok(())
    }

    /// Stops the cleaning thread and clears the entire cache data.
    pub fn stop(&mut self) -> Result<(), CacheError> {
        self.data.try_lock()?.clear();
        self.running.swap(false, Ordering::Relaxed);
        Ok(())
    }

    /// Checks if the cleaning thread is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Returns a list of all the values in the cache
    pub fn list(&self) -> Result<Vec<V>, CacheError> {
        Ok(self
            .data
            .try_lock()?
            .iter()
            .map(|(_, entry)| entry.data.clone())
            .collect())
    }
}

/// Drop implemented to make sure the thread is terminated correctly
impl<K, V> Drop for ConcurrentTTLCache<K, V> {
    fn drop(&mut self) {
        self.running.swap(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestCache = ConcurrentTTLCache<String, u32>;

    fn new_no_clear_cache() -> TestCache {
        let mut tc = TestCache::new(5000);
        tc.stop_clear().unwrap();
        tc
    }

    #[test]
    fn test_get_put() {
        let mut c = new_no_clear_cache();

        c.put(String::from("test"), 1).unwrap();
        let v = c.get(&String::from("test")).unwrap();

        assert_eq!(v, 1);
    }

    #[test]
    fn test_list() {
        let mut c = new_no_clear_cache();
        c.put(String::from("test1"), 1).unwrap();
        c.put(String::from("test2"), 2).unwrap();

        let list = c.list().unwrap();
        assert!(list.contains(&1)); // Order of insertion is not guaranteed with a HashMap
        assert!(list.contains(&2)); // so we can't check the Vec's values by index or order
        assert!(list.len() == 2);
    }

    #[test]
    fn test_clear() {
        let mut c: ConcurrentTTLCache<String, u32> = ConcurrentTTLCache::new(1000);

        c.put(String::from("test1"), 1).unwrap();
        assert!(c.list().unwrap().len() > 0);

        c.clear().unwrap();
        assert!(c.list().unwrap().len() == 0);
        assert!(c.is_running());
    }

    #[test]
    fn test_stop() {
        let mut c: ConcurrentTTLCache<String, u32> = ConcurrentTTLCache::new(1000);

        c.put(String::from("test1"), 1).unwrap();

        assert!(c.list().unwrap().len() > 0);

        c.stop().unwrap();
        assert!(c.list().unwrap().len() == 0);
        assert!(!c.is_running());
    }

    #[test]
    fn test_stop_clear() {
        let mut c: ConcurrentTTLCache<String, u32> = ConcurrentTTLCache::new(1000);

        c.put(String::from("test1"), 1).unwrap();
        c.stop_clear().unwrap();
        assert!(c.list().unwrap().len() > 0);
        assert!(!c.is_running());
    }
}
