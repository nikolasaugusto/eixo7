document.getElementById("priceForm").addEventListener("submit", function(e) {
  e.preventDefault();

  // Cria o objeto com os dados dos inputs
  const formData = {
    "Área_útil": parseFloat(document.getElementById("areaUtil").value),
    "Quartos": parseFloat(document.getElementById("quartos").value),
    "Banheiros": parseFloat(document.getElementById("banheiros").value),
    "Vagas_na_garagem": parseFloat(document.getElementById("vagasGaragem").value),
    "IPTU": parseFloat(document.getElementById("iptu").value),
    "Condomínio": parseFloat(document.getElementById("Condominio").value) || 0,
    "Tipo": document.getElementById("selectTipo").value
  };

  // Mapeia os checkboxes para os nomes de chaves esperados pela API
  const checkboxMapping = {
    "academia": "Academia",
    "elevador": "Elevador",
    "permitidoAnimais": "Permitido_animais",
    "piscina": "Piscina",
    "portaria": "Portaria",
    "salãoFestas": "Salão_de_festas",
    "portaoEletronico": "Portão_eletrônico",
    "areaMurada": "Área_murada",
    "areaServico": "Área_de_serviço",
    "armariosCozinha": "Armários_na_cozinha",
    "armariosQuarto": "Armários_no_quarto",
    "churrasqueira": "Churrasqueira",
    "mobiliado": "Mobiliado",
    "quartoServico": "Quarto_de_serviço",
    "arCondicionado": "Ar_condicionado",
    "porteiro24h": "Porteiro_24h",
    "varanda": "Varanda"
  };

  document.querySelectorAll('input[name="caracteristicas"]').forEach(checkbox => {
    if (checkboxMapping[checkbox.id]) {
      formData[checkboxMapping[checkbox.id]] = checkbox.checked ? 1 : 0;
    }
  });

  console.log("Dados a serem enviados:", formData);

  // Lista de endpoints a serem consultados com os respectivos labels
  const endpoints = [
    { label: "Random Forest", url: "http://127.0.0.1:8000/predict" },
    { label: "Linear Regression", url: "http://127.0.0.1:8000/predict/linear" },
    { label: "Gradient Boosting", url: "http://127.0.0.1:8000/predict/gradient" }
  ];

  // Limpa a área de resultados antes de inserir os novos resultados
  const resultadosContainer = document.getElementById("resultados");
  resultadosContainer.innerHTML = "";

  // Para cada endpoint, faz a requisição e exibe o resultado
  endpoints.forEach(model => {
    fetch(model.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(formData)
    })
    .then(response => {
      if (!response.ok) {
        throw new Error("Erro na requisição para " + model.label);
      }
      return response.json();
    })
    .then(data => {
      // Cria um elemento para exibir a resposta de cada modelo
      const resultDiv = document.createElement("div");
      resultDiv.className = "mt-3 alert alert-info";
      resultDiv.innerHTML = `<h5>${data.modelo}</h5>
                             <p><strong>Preço Previsto: R$${data.preco_previsto}</strong></p>`;
      resultadosContainer.appendChild(resultDiv);
    })
    .catch(error => {
      console.error("Erro:", error);
      const errorDiv = document.createElement("div");
      errorDiv.className = "mt-3 alert alert-danger";
      errorDiv.innerHTML = `<h5>${model.label}</h5><p>Ocorreu um erro: ${error.message}</p>`;
      resultadosContainer.appendChild(errorDiv);
    });
  });
});
