document.getElementById('nutrition-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert("Harap unggah gambar terlebih dahulu.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/vision/nutrition', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("Gagal memproses gambar.");
        }

        const data = await response.json();
        document.getElementById('calories').textContent = `${data.calories.toFixed(2)} kcal`;
        document.getElementById('protein').textContent = `${data.protein.toFixed(2)} g`;
        document.getElementById('fat').textContent = `${data.fat.toFixed(2)} g`;
        document.getElementById('carbohydrates').textContent = `${data.carbohydrates.toFixed(2)} g`;

        document.getElementById('nutrition-result').classList.remove('hidden');
    } catch (error) {
        console.error("Error:", error);
        alert("Terjadi kesalahan saat memproses gambar.");
    }
});
