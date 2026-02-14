# Filters (A/B/C)

This project defines three **entry filters**.

Current default/assumption (v2):
- **Filter A (PL) = ON**
- **Filter B (Ticker cycle) = ON (always applied in standard runs)**
- Filter C (Market cycle) remains optional (requires market index CSV).

- **Filter A (PL filter)**  
  Prevents immediately flipping direction after a profitable trade.  
  If the most recent closed trade was profitable, the next **opposite-direction** entry is blocked.

- **Filter B (Ticker Moving-Average Cycle filter)** *(standard: always ON)*  
  Uses EMA(5/20/40) phase classification (Kojiro-style 6-phase cycle).  
  - **Long allowed:** phases **6, 1, 2**  
  - **Short allowed:** phases **3, 4, 5**

- **Filter C (Market Moving-Average Cycle filter)**  
  Same as Filter B but computed on the **market index** (e.g., KOSPI200 futures/index CSV).  
  - **Long allowed:** phases **6, 1, 2**  
  - **Short allowed:** phases **3, 4, 5**

In older combo tests, filter sets were encoded as codes like `NONE`, `B`, `BC`.

In the current standard configuration, **Filter B is not a sweep dimension**.
