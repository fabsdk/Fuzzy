import skfuzzy as fz
from skfuzzy import control as ctr
import numpy as np

class Fuzzy():
  def __init__(self):
    self.text = None
    self.point = 0
    # aqui está o banco de palavras positivas, negativas e intensificadores gerado com o chatGPT
    self.posWords = ["excepcional","ótimo", "excelente", "magnífico", "fantástico", "maravilhoso", "feliz", "espetacular", "incrível", "surpreendente", "admirável"]
    self.negWords = ["repugnante", "péssimo", "horrível", "terrível", "desagradável","frustrante", "decepcionante", "triste", "deplorável", "desastroso", "desanimador", "desprezível"]
    self.intenWords  = ["tão","muito", "altamente", "bastante", "extremamente", "demais", "incrivelmente", "realmente", "completamente","profundamente"]
    self.negatWords = ["não", "jamais", "nenhum", "nem", "nada", "nunca", "falso", "incorreto", "impossível"]
    
    # Aqui estamos definindo as variáveis antecedentes do sistema fuzzy
    # Definindo seus universos de discurso
    # Deste modo o sistema consegue fazer uma analise adquadada dos sentimentos
    self.wordPos = ctr.Antecedent(np.arange(0, 2, 1), "FP")
    self.wordNeg = ctr.Antecedent(np.arange(0, 2, 1), "FN")
    self.wordInte = ctr.Antecedent(np.arange(0, 2, 1), "I")
    self.wordNegat = ctr.Antecedent(np.arange(0, 2, 1), "N")

    # Aqui estamos definindo os conjuntos fuzzy para cada variável antecedente
    # nao representa o valor 0 e sim o valor 1 (ou sejá, há ou não há a palavra na frase)
    self.wordPos.automf(names=["nao", "sim"])
    self.wordNeg.automf(names=["nao", "sim"])
    self.wordInte.automf(names=["nao", "sim"])
    self.wordNegat.automf(names=["nao", "sim"])
    
    # A variável consequente representa o resultado ou a saída do sistema fuzzy
    # Os valores vão de 0 a 10 onde cada valor representa um grau de sentimento
    self.emotion = ctr.Consequent(np.arange(0, 11, 1), "sentimento")

    
    # Cada variável define um conjunto fuzzy diferente para representar um possível estado de sentimento, 
    # utilizada uma função de pertinência triangular cujos valores máximos são alcançados em 5,7 e 9 no caso do positivo
    self.emotion["positivo"] =  fz.trimf(self.emotion.universe, [5, 7, 9])
    self.emotion["positivasso"] = fz.trimf(self.emotion.universe, [7, 10, 10])
    self.emotion["neutro"] =    fz.trimf(self.emotion.universe, [3, 5, 7])
    self.emotion["negativasso"] = fz.trimf(self.emotion.universe, [0, 0, 3])
    self.emotion["negativo"] =  fz.trimf(self.emotion.universe, [1, 3, 5])
    
    
    regras = []

    # Aqui estamos criando os possiveis cenários positivos
    regras.append(ctr.Rule(self.wordPos["sim"] & self.wordNeg["nao"], self.emotion["positivo"]))
    regras.append(ctr.Rule(self.wordPos["sim"] & self.wordInte["sim"], self.emotion["positivasso"]))
    regras.append(ctr.Rule(self.wordNeg["sim"] & self.wordNegat["sim"], self.emotion["positivo"]))
    regras.append(ctr.Rule(self.wordPos["sim"], self.emotion["positivo"]))


    # Aqui estamos criando os possiveis cenários neutros
    regras.append(ctr.Rule(self.wordNeg["nao"] & self.wordPos["nao"], self.emotion["neutro"]))
    regras.append(ctr.Rule(self.wordNeg["sim"] & self.wordPos["sim"], self.emotion["neutro"]))

    # Aqui estamos criando os possiveis cenários negativos
    regras.append(ctr.Rule(self.wordNeg["sim"], self.emotion["negativo"]))
    regras.append(ctr.Rule(self.wordNeg["sim"] & self.wordPos["nao"], self.emotion["negativo"]))
    regras.append(ctr.Rule(self.wordPos["sim"] & self.wordNegat["sim"], self.emotion["negativo"]))
    regras.append(ctr.Rule(self.wordNeg["sim"] & self.wordInte["sim"], self.emotion["negativasso"]))
    
    self.ctrlSys = ctr.ControlSystem(regras)
    self.simulador = ctr.ControlSystemSimulation(self.ctrlSys)

   # Nessa função é carregado a frase e é feita a verificação de cada palavra
  def obtendoFrase(self, frase):
        separarFrase = frase.split(" ")
        self.frase = frase
        # Aqui é feita a verificação de cada palavra
        self.simulador.input["FP"] = 1 if len([palavra for palavra in separarFrase if palavra in self.posWords]) > 0 else 0
        self.simulador.input["FN"] = 1 if len([palavra for palavra in separarFrase if palavra in self.negWords]) > 0 else 0
        self.simulador.input["I"] = 1 if len([palavra for palavra in separarFrase if palavra in self.intenWords]) > 0 else 0
        self.simulador.input["N"] = 1 if len([palavra for palavra in separarFrase if palavra in self.negatWords]) > 0 else 0
  
  
  # Obtendo grau de verdade
  def analisar(self):
        # Ativa o processo de inferência fuzzy.
        self.simulador.compute()
        self.grau_sentimento = self.simulador.output["sentimento"]

  # Mapeia o valor de saída para uma categoria de sentimento específica.
  def polaridadeSentimento(self):
      if self.grau_sentimento < 2.3:
          msg = "Negativassa"
      elif self.grau_sentimento > 2.3 and self.grau_sentimento < 3.5:
          msg = "Negativa"
      elif self.grau_sentimento > 3.5 and self.grau_sentimento < 5.1:
          msg = "Neutra"
      elif self.grau_sentimento > 5.1 and self.grau_sentimento < 7.5:
          msg = "Positiva"
      else:
          msg = "Positivassa"

      return msg

if "__main__" == __name__:
    
    # Frases para teste geradas com o chatGPT
    frases = ["Este filme é realmente excelente",
              "Fiquei extremamente feliz com o resultado",
              "Infelizmente o serviço foi péssimo",
              "O concerto foi maravilhoso",
              "Esse programa de TV é horrível",
              "Ele é um ator incrível",
              "Nunca vi um filme tão desastroso quanto esse",
              "Foi uma experiência decepcionante",
              "A qualidade do produto é realmente excepcional",
              "Não há nada de positivo a dizer sobre esse evento",
              "Esse relógio é extremamente horrível"]
    
    print("\n😁  Olá, seja bem vindo ao meu sistema fuzzy de análise de sentimentos 😭\n")

    #verifica cada frase na lista
    for frase in frases:
      #chama a classe fuzzy
      fuzzy = Fuzzy()
      #obtem as frases
      fuzzy.obtendoFrase(frase)
      #analisa as frases
      fuzzy.analisar()
      #imprime o resultado
      print(f"✏️  {frase} ➜  De acordo com o meu banco essa frase é {fuzzy.polaridadeSentimento()}")