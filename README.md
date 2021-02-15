# Sirius Resim Sınıflandırıcı

Türkçe detaylı açıklamalar için, SiriusRapor.pdf'i inceleyebilirsiniz.

Raporda bulunmayan özellikler:
- Epoch limiti belirlemek adına ön işleme sayfasına TextBox eklenmiştir. Bu TextBox
ile Epoch adedini belirlemeniz zorunludur. 5-10 arası vermeniz tavsiye edilir.

# Sirius Image Classifier

<b>You can create new project with "Yeni Proje" button.</b><br>
<b>You can open an existing project with "Proje Aç" button.</b>

<b>In PreProcess screen, you can make Data Augmentation with "Veri arttır" button</b><br>
<b>
You can choose dataset for training-test in first Combobox. If you choose 
"Orjinal görüntüler", program is ignore augmentated images. If you choose
"Arttırılmış görüntüler + Orjinal görüntüler", its learn both of original images & 
augmentated images. 
</b><br>

<b>You can choose transfer learning base model with second combobox.</b><br><br>
<b>You need to give epoch limit with textbox. Other way, your model is not gonna work.
We don't recommend working with large epoch limits. 5-10 is the better for this model.
You can change model architecture an limits by the way.</b> <br>

<b>When your classification ends, you can take a look to classification report. You can
choose an image with "Resim Seç" button and classify that with "Sınıflandır" button.</b>