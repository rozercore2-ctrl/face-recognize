from register import register_face  # fungsi registrasi
from train import train_model, load_label_dict  # fungsi training
from recognize_with_gsheets import recognize_face  # recognition yg sudah include upload sheets

def main():
    label_dict = None
    while True:
        print("\n=== Menu Face Recognition ===")
        print("1. Registrasi")
        print("2. Training")
        print("3. Recognition (upload ke Google Sheets)")
        print("4. Keluar")
        choice = input("Pilih menu: ").strip()

        if choice == '1':
            register_face()
        elif choice == '2':
            label_dict = train_model()
        elif choice == '3':
            # Load label_dict kalau belum ada
            if label_dict is None:
                label_dict = load_label_dict()
                if label_dict is None:
                    print("Silakan lakukan training terlebih dahulu.")
                    continue
            recognize_face()  # langsung jalankan recognize yang upload ke Google Sheets
        elif choice == '4':
            print("Keluar program.")
            break
        else:
            print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()
