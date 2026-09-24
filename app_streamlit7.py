import streamlit as st  # streamlit ile web arayüzü oluştur
import requests  # fast api ye http isteği gönder

# başlık ve açıklama
st.title("Trafik Hacmi Tahmini")
st.markdown("""
    Son 24 saate ait sıcaklık, yapmur, kar, bulutluluk oranı, saat, gün ve ay bilgilerini girin.
    Her saat için varsayılan değerler dolu olarak gelir. Dilerseniz değiştirebilirsiniz. **Tahmin Et** butonuna basarak sonucu görebilirsiniz.
""")

# 24 saatlik girdi verisi hazırlama
# her saat için [temp, rain_1h, snow_1h, clouds_all, hour, dayofweek, month]
# kullanıcının girdiği veriler sequence_input listesinde toplanır

sequence_input = []  # 24x7

# form bloğu başlat (girdileri form içerisinde toplar)
with st.form("manual_input_form"):
    st.subheader("24 Saatlik Girdi Verisi")

    # her saat için 7 özellik girilecek
    for i in range(24):
        st.markdown(f"**🕐 Saat {i}**")

        # 7 sütun halinde grid tasarımı
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        #her bir özellik için kllanıcıdan input alalım (ayrıca biz bunları varsayılan değerler ile dolduralım)
        temp = col1.number_input("Temp (K)", value = 290.0, min_value = 270.0, max_value = 330.0, step = 0.1, key = f"temp_{i}")
        rain = col2.number_input("Rain (mm)", value = 0.0, min_value=0.0, max_value=100.0, step = 0.1, key = f"rain_{i}")
        snow = col3.number_input("Snow (mm)", value = 0.0, min_value=0.0, max_value=100.0, step=0.1, key = f"snow_{i}")
        clouds = col4.slider("Clouds (%)", value = 40, min_value = 0, max_value=100, key = f"clouds_{i}")
        hour = i  # 0 dan 23 e kadar otomatik olarak verilir
        dayofweek = col6.selectbox("Gün", options=list(range(7)), index = 1, key = f"day_{i}")
        month = col7.selectbox("Ay", options=list(range(1,13)), index = 9, key = f"month_{i}")

        # kullanıcının girdiği 7 özelliği bir listede toplatalım
        sequence_input.append([temp, rain, snow, clouds, hour, dayofweek, month])

    # forma gönderme butonu ekle
    submitted = st.form_submit_button("🔮 Tahmin Et")

    #tahmin et butonuna basıldığında FastAPI  ye POST isteği gönder
    #FastAPI servisinden tahmin sonucu alınır ve ekrana yazılır

    if submitted:
        try:
            #fast api servisi çalışıyorsa url e post isteği gönder
            API_URL = "http://127.0.0.1:8000/predict"

            response = requests.post(API_URL, json={"sequence": sequence_input}) #json formatında post isteği gönder

            if response.status_code == 200:
                result = response.json()
                tahmin = result["tahmin_edilen_trafik_hacmi"]

                #basarılı tahmini kullanıcıya göster
                st.success(f"Tahmin Edilen Trafik Hacmi: {int (tahmin)} araç/saat")
            else:
                st.error(f"API isteği başarısız. Kod :{response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error("FastAPI sunucuna bağlanamadı.")
            st.exception(e)        

  


