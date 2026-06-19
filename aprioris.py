import pandas as pd
from mpi4py import MPI
from collections import Counter
import sys
import itertools # Ditambahkan untuk mempermudah kombinasi aturan

# Inisialisasi MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def print_ui(langkah, judul):
    if rank == 0:
        print(f"\n{'='*75}", flush=True)
        print(f"[LANGKAH {langkah}] {judul}", flush=True)
        print(f"{'='*75}", flush=True)

# Fungsi untuk menghasilkan kombinasi kandidat (Ck) berdasarkan itemset yang lolos sebelumnya (L_k-1)
def generate_candidates(L_prev, k):
    candidates = set()
    n = len(L_prev)
    for i in range(n):
        for j in range(i + 1, n):
            set1 = list(L_prev[i])
            set2 = list(L_prev[j])
            set1.sort()
            set2.sort()
            
            if set1[:k-2] == set2[:k-2]:
                new_itemset = tuple(sorted(set(set1) | set(set2)))
                if len(new_itemset) == k:
                    candidates.add(new_itemset)
    return list(candidates)

chunks_dataset = None
dataset_bagian_saya = None
minimal_muncul = 0
total_transaksi = 0
persen_minimum = 3 # Ubah persentase minimum support di sini

# Parameter baru untuk aturan asosiasi
min_confidence = 0.5 # Contoh: minimal 50% tingkat keyakinan

# Sinkronisasi semua prosesor sebelum mulai menghitung waktu
comm.Barrier()
if rank == 0:
    start_time = MPI.Wtime() 

if rank == 0:
    print_ui("1", "MEMBACA DATASET & MENGUBAH KE FORMAT TRANSAKSI")
    nama_file = 'Groceries_dataset_massive.csv'
    
    jumlah_baris = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    try:
        if jumlah_baris:
            df = pd.read_csv(nama_file, nrows=jumlah_baris)
            print(f"-> Berhasil membaca {jumlah_baris} baris dari file '{nama_file}'.")
        else:
            df = pd.read_csv(nama_file)
            print(f"-> Berhasil membaca file '{nama_file}' secara utuh.")
            
    except FileNotFoundError:
        print(f"[Error] File {nama_file} tidak ditemukan di direktori!")
        comm.Abort()
        sys.exit()

    kolom_id = 'Member_number' 
    kolom_item = 'itemDescription'

    dataset_utuh = df.groupby(kolom_id)[kolom_item].apply(lambda x: set(x)).values.tolist()
    item_unik = list(df[kolom_item].unique())
    
    total_transaksi = len(dataset_utuh)
    minimal_muncul = (persen_minimum / 100) * total_transaksi

    print(f"-> Ditemukan total {total_transaksi} transaksi unik.")
    print(f"-> Ditemukan {len(item_unik)} jenis barang unik di seluruh dataset.")
    print(f"-> Batas minimum kemunculan ({persen_minimum}%): {minimal_muncul:.2f} transaksi.")

    print_ui("2", f"DISTRIBUSI DATA KE {size} PROSESOR (SCATTER)")
    print("-> Memecah dataset transaksi agar dikerjakan bersama-sama...")
    
    chunks_dataset = [[] for _ in range(size)]
    for i, transaksi in enumerate(dataset_utuh):
        chunks_dataset[i % size].append(transaksi)

minimal_muncul = comm.bcast(minimal_muncul, root=0)

# Scatter dataset transaksi ke masing-masing prosesor
dataset_bagian_saya = comm.scatter(chunks_dataset, root=0)

t_awal = MPI.Wtime()
while MPI.Wtime() - t_awal < (rank * 0.1): pass 
print(f"   [Prosesor {rank}] Menerima {len(dataset_bagian_saya)} transaksi untuk diperiksa.", flush=True)

comm.Barrier()

k = 1
L_prev = [] 

# KAMUS BARU: Untuk menyimpan semua itemset yang lolos beserta support count-nya (Penting untuk kalkulasi Aturan)
all_frequent_itemsets_support = {} 

while True:
    kandidat_utuh = []
    
    if rank == 0:
        print_ui(f"3.{k}", f"PEMBENTUKAN KANDIDAT ITEMSET (TAHAP C{k})")
        if k == 1:
            kandidat_utuh = [(item,) for item in item_unik if pd.notna(item)]
        else:
            kandidat_utuh = generate_candidates(L_prev, k)
            
        print(f"-> Total Kandidat (C{k}) yang akan dicek: {len(kandidat_utuh)} kombinasi.")

    kandidat_utuh = comm.bcast(kandidat_utuh, root=0)
    
    if len(kandidat_utuh) == 0:
        if rank == 0:
            print(f"-> Tidak ada lagi kandidat yang bisa dibentuk untuk level {k}. Proses iterasi berhenti.")
        break

    # PERHITUNGAN LOKAL
    hasil_hitung_parsial = Counter()
    for barang in kandidat_utuh:
        jumlah = 0
        set_barang = set(barang)
        for transaksi in dataset_bagian_saya:
            if set_barang.issubset(transaksi):
                jumlah += 1
        if jumlah > 0:
            hasil_hitung_parsial[barang] = jumlah

    # PENGGABUNGAN HASIL
    hasil_akhir = comm.reduce(hasil_hitung_parsial, op=MPI.SUM, root=0)

    L_current = []
    if rank == 0:
        print(f"\n[Tahap L{k}] Evaluasi Minimum Support (>= {minimal_muncul:.2f})")
        
        lolos = []
        tidak_lolos = []
        
        for barang in kandidat_utuh:
            jumlah = hasil_akhir.get(barang, 0)
            if jumlah >= minimal_muncul:
                L_current.append(barang)
                lolos.append(f"   [v] {barang} : {jumlah} kali")
                # SIMPAN KE KAMUS GLOBAL (Rank 0)
                all_frequent_itemsets_support[barang] = jumlah
            else:
                tidak_lolos.append(f"   [x] {barang} : {jumlah} kali")
                
        print(f"-> Ditemukan {len(L_current)} kombinasi yang LOLOS (Frequent {k}-Itemsets):")
        for teks in lolos[:10]: 
            print(teks)
        if len(lolos) > 10:
            print(f"   ... (dan {len(lolos) - 10} kombinasi lolos lainnya)")

    L_current = comm.bcast(L_current, root=0)
    L_prev = L_current
    
    if len(L_prev) == 0:
        if rank == 0:
            print(f"-> Tidak ada frequent itemset yang lolos di level {k}. Proses iterasi berhenti.")
        break
        
    k += 1 
    comm.Barrier()

comm.Barrier()

# =============================================================================
# TAHAP BARU: PEMBENTUKAN ATURAN ASOSIASI (CONFIDENCE & LIFT RATIO)
# =============================================================================
if rank == 0:
    print_ui("4", "PEMBENTUKAN ATURAN ASOSIASI (CONFIDENCE & LIFT RATIO)")
    print(f"-> Memproses itemset dengan ukuran >= 2 untuk membuat aturan (Jika ada)...")
    
    aturan_ditemukan = 0
    
    # Looping setiap itemset yang lolos support dan ukurannya >= 2
    for itemset, support_count_gabungan in all_frequent_itemsets_support.items():
        if len(itemset) < 2:
            continue
            
        # Cari semua subset sejati dari itemset tersebut untuk dijadikan 'Antecedent' (Jika membeli A...)
        # Misal itemset: ('roti', 'susu'), subset-nya: ('roti',), ('susu',)
        for r in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, r):
                # Consequent (Maka membeli B...) adalah sisa item di dalam itemset
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                
                # Ambil nilai support count masing-masing dari kamus histori
                support_count_A = all_frequent_itemsets_support.get(antecedent, 0)
                support_count_B = all_frequent_itemsets_support.get(consequent, 0)
                
                if support_count_A > 0:
                    # 1. RUMUS CONFIDENCE: P(B|A) = Support(A U B) / Support(A)
                    confidence = support_count_gabungan / support_count_A
                    
                    # Cek apakah memenuhi batas minimum confidence
                    if confidence >= min_confidence:
                        # 2. RUMUS LIFT RATIO: Confidence(A -> B) / Support_Persen(B)
                        # Atau bentuk matematis sederhananya menggunakan counts:
                        # Lift = (Count(A U B) * Total_Transaksi) / (Count(A) * Count(B))
                        lift_ratio = (support_count_gabungan * total_transaksi) / (support_count_A * support_count_B)
                        
                        aturan_ditemukan += 1
                        print(f"Aturan #{aturan_ditemukan}:")
                        print(f"   Jika membeli {antecedent} -> Maka membeli {consequent}")
                        print(f"   - Support Gabungan : {support_count_gabungan} kali ({(support_count_gabungan/total_transaksi)*100:.2f}%)")
                        print(f"   - Confidence       : {confidence*100:.2f}%")
                        print(f"   - Lift Ratio       : {lift_ratio:.4f} " + 
                              ("(Hubungan Positif / Valid)" if lift_ratio > 1 else "(Hubungan Kebetulan / Negatif)"))
                        print("-" * 50)
                        
    if aturan_ditemukan == 0:
        print("-> Tidak ada aturan yang memenuhi batas minimum confidence yang ditentukan.")

if rank == 0:
    end_time = MPI.Wtime() 
    waktu_eksekusi = end_time - start_time
    
    print_ui("SELESAI", "PROSES APRIORI PARALEL BERAKHIR")
    print(f"[*] Frekuensi maksimal yang dicapai : {k - 1}-itemset")
    print(f"[*] Waktu Pemrosesan Total        : {waktu_eksekusi:.4f} detik")
    print(f"{'='*75}\n")