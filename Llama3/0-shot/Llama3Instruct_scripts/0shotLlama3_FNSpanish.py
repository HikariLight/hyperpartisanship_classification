import pandas as pd
import transformers
from transformers import LlamaTokenizer,pipeline, LlamaForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, AutoModelForSequenceClassification
import os
from trl import SFTTrainer, SFTConfig
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
import evaluate 
import torch
import re
import time
import random
from sklearn.metrics import precision_recall_fscore_support, classification_report
import json

random.seed(42)

start_time = time.time()
configuration = "5-shot"
n_example = 5
temp = 0.1
sys_ans = []

tsv_directory = ''
filename = 'FakeNewsCorpusSpanish.csv'
df_path = os.path.join(tsv_directory, filename)
df = pd.read_csv(df_path)
#df = df.groupby('Category').head(n_example)
df['Category'] = df['Category'].map({'True': 0, 'Fake': 1})
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
lora_config = LoraConfig(
    r = 8, # the dimension of the low-rank matrices
    lora_alpha = 16, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'o_proj', 'k_proj', 'v_proj'],
    lora_dropout = 0.1, # dropout probability of the LoRA layers
    bias = 'none', #wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

device = torch.device('cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map=device, 
                                             quantization_config=quantization_config,
                                             torch_dtype=torch.float16)


prompt = """
### Instruction:
Dada una noticia, determina si es real o falsa.
La salida consiste en un solo número entero con este formato: 'entero'. Si la noticia es falsa, la salida será '1', si es real, '0'.
#Ejemplo: 'Alerta: pretenden aprobar libros escolares con contenido sesgado sobre el conflicto armado interno\nSigue la campaña negacionista del fujimorismo. Otra vez desde el Congreso, la bancada Fuerza Popular (FP) presionó para que se revisen los textos escolares y sus contenidos sobre la época del terrorismo, según denuncia la Coordinadora Nacional de Derechos Humanos (Cnddhh).\nDe acuerdo a esta organización, el Ministerio de Educación (Minedu) estaría a punto de aprobar libros de primero, segundo y tercero de secundaria con contenidos sesgados sobre el periodo del conflicto armado interno, faltando así a la verdad y a la memoria de los cientos de miles de víctimas de la violencia y el terror desatadas entre los años *NUMBER* y *NUMBER* en Perú.\nLa Cnddhh denunció que esto sería el resultado de un convenio suscrito, en setiembre del *NUMBER*, entre el entonces ministro de Educación, Idel Vexler, y el presidente del Congreso, el fujimorista Luis Galarreta, para la conformación de una comisión que incluía a personajes como el exvicepresidente de Alberto Fujimori, Francisco Tudela, y el vicealmirante en retiro, implicado en la matanza de El Frontón, Luis Giampietri.\nEn este proceso de revisión, se habrían realizado talleres en los que participaron congresistas y asesores de FP y miembros en retiro de las Fuerzas Armadas, en lugar de instituciones y especialistas expertos en asuntos pedagógicos y derechos humanos. En este sentido, la Cnddhh advirtió que esta es una modalidad de injerencia del Congreso en competencias que corresponden al Poder Ejecutivo.\nLa Cnddhh sostuvo que esto tiene la finalidad de negar los hechos ocurridos, así como "las responsabilidades de los distintos actores que cometieron graves crímenes contra la humanidad y sometieron al país al miedo y al terror, incluyendo la responsabilidad del propio Estado". Este es el pronunciamiento de la Cnddhh:\nSOBRE EL MINEDU Y LOS TEXTOS ESCOLARES:\nLAS POLÍTICAS EDUCATIVAS NO PUEDEN DESCONOCER EL PASADO DEL PAÍS\nPronunciamiento\nLa Coordinadora Nacional de Derechos Humanos (CNDDHH), frente a la propuesta del Ministerio de Educación (MINEDU) de contenidos para la enseñanza sobre el terrorismo en el Perú en los textos escolares, que se pretenden validar e imprimir con extrema celeridad, señala y alerta a la ciudadanía con carácter de urgencia sobre lo siguiente:\n*NUMBER*. El actual Congreso de la República y su mayoría vienen ejerciendo presiones sobre el poder ejecutivo y demás entidades del Estado para favorecer sus intereses particulares, que resultan lesivos al país y sus mayorías ciudadanas. En este contexto, ha desplegado una agresiva campaña que pretende negar una visión comprehensiva del período del conflicto armado interno (CAI), así como las responsabilidades de los distintos actores que cometieron graves crímenes contra la humanidad y sometieron al país al miedo y al terror, incluyendo la responsabilidad del propio Estado.\n*NUMBER*. La campaña del negacionismo se ha propuesto imponer a través del MINEDU una visión sesgada de la historia y de los hechos del CAI, afectando así el derecho a la verdad que tiene la ciudadanía, en particular las niñas, niños y adolescentes en formación escolar. Para ello, el *NUMBER* de septiembre del *NUMBER* el Congreso y el MINEDU suscribieron un convenio para "reforzar la educación cívica y democrática en los colegios" y anunciaron la conformación de una "Comisión" para este fin. Abriendo así una peligrosa puerta para la injerencia del Congreso sobre competencias que corresponden al gobierno.\n*NUMBER*. En esta orientación, la actual gestión del MINEDU ha realizado tres (*NUMBER*) talleres de consulta para validar los contenidos de los textos escolares, priorizando la participación de congresistas y asesores de Fuerza Popular y la de miembros en situación de retiro de las FFAA. El proceso de validación adquiere así un carácter político que se impone sobre los criterios técnico, pedagógico y especializado que deben garantizar el derecho a la educación. Este proceso omite además criterios metodológicos claros, así como la participación amplia de especialistas en la materia.\n*NUMBER*. La política educativa del país debe ser leal a nuestra historia reciente, a los principios democráticos y al respeto profundo a los derechos fundamentales de las personas. Como lo advierte el Consejo de Derechos Humanos de las Naciones Unidas, en su estrategia global para hacer frente a las condiciones que propician la propagación del terrorismo, las "medidas eficaces contra el terrorismo y la protección de los derechos humanos no son objetivos contrapuestos".\n*NUMBER*. El derecho a la verdad es un derecho fundamental individual y colectivo reconocido por nuestro Tribunal Constitucional y la Corte Interamericana de Derechos Humanos, que exige a los Estados promover en sus políticas, más aún de educación, objetividad sobre qué pasó en el país y cuáles fueron sus causas. En este sentido, el MINEDU no puede someterse a las presiones del Congreso y menos hacerse cómplice del negacionismo.\nBajo estas consideraciones, la CNDDHH exhorta al MINEDU y exige al Estado peruano a promover un diálogo amplio y abierto sobre los contenidos y fines de los mencionados textos escolares, conducente a garantizar a las futuras generaciones el derecho a una política educativa y de memoria que respete los principios democráticos y el derecho fundamental a la verdad. pretenden aprobar libros escolares con contenido sesgado sobre el conflicto armado interno', '1'
#Ejemplo: 'Madonna será la nueva imagen del Chocolate Abuelita\nA partir del próximo mes, habrá un cambio drástico en uno de los productos más emblemáticos de la cocina mexicana, el Chocolate Abuelita, el cual existe desde *NUMBER*, teniendo como imagen a doña Sara García desde *NUMBER*\n"Desde que el producto pertenecía a La Azteca, se ha utilizado la foto de Sara García, que durante muchos años nos funcionó perfectamente", declaró en entrevista con El la señorita Becher Schokoladenmilch, vocera de Nestlé para México. "Sin embargo, la imagen de las abuelas modernas ha cambiado notablemente. Y si son de la tercera edad, ya no podemos llamarlas ancianas, y eso es importante".\nLa vocera agregó que a partir del próximo mes, la cantante Madonna será la imagen del producto, que se espera se pueda comercializar a nivel mundial. "Madonna representa a la nueva mujer de edad para ser abuela que sigue siendo activa e independiente. Además, ella es más conocida que la señora García, que a pesar de su notable carrera, no es muy identificada fuera de México. Esta es, sin duda, nuestra mejor estrategia comercial en años".\nLa ejecutiva nos mostró lo que sería el primer comercial, de Madonna en concierto, anunciando el producto, aunque desafortunadamente, no podrá presentarse al público hasta que la noticia se haga oficial. Madonna será la nueva imagen del Chocolate Abuelita', '1'
#Ejemplo: 'La palabra "haiga", aceptada por la RAE La Real Academia de la Lengua (RAE), ha aceptado el uso de "HAIGA", para su utilización en las tres personas del singular del presente del subjuntivo del verbo hacer, aunque asegura que la forma más recomendable en la lengua culta para este tiempo, sigue siendo "haya".\nAsí lo han confirmado fuentes de la RAE, que explican que este cambio ha sido propuesto y aprobado por el pleno de la Academia de la Lengua, tras la extendida utilización por todo el territorio nacional, sobre todo, empleado por personas carentes de estudios o con estudios básicos de graduado escolar. Ya no será objeto de burla ese compañero que a diario repite aquello de "Mientras que haiga faena, no podemos quejarnos" o esa abuela que repite aquello de "El que haiga sacao los juguetes, que los recoja".\nEntre otras palabras novedosas que ha aceptado la RAE, contamos también con "Descambiar", significa deshacer un cambio, por ejemplo "devolver la compra". Visto lo visto, nadie apostaría que la palabra "follamigos" sea la siguiente de la lista. La palabra "haiga", aceptada por la RAE', '1'
#Ejemplo: 'YORDI ROSADO ESCRIBIRÁ Y DISEÑARÁ LOS NUEVOS LIBROS DE TEXTO DE LA SEP PARA HACERLOS MÁS ATRACTIVOS\nMéxico.- El director de la Secretaría de Educación Pública, Aurelio Nuño, informó que el dramaturgo y conductor Yordi Rosado será el encargado de redactar los nuevos libros de texto que se reparten en todas las escuelas del país, y que a partir del próximo ciclo escolar dejarán de ser gratuitos y tendrán costo.\nAurelio señala que decidió contratar a Yordi para escribir los manuales con el fin de hacer "más atractivos los textos" y que los alumnos se interesen más por aprender, por lo que los libros dejarán de tener un lenguaje aburrido y serán presentados con un idioma "más fresco, juvenil y moderno":\n"Desafortunadamente el gobierno ya no tiene los recursos para seguir regalando los libros, por lo que a partir del próximo ciclo escolar los padres de familia deberán hacer un pequeño sacrificio y pagar una módica cantidad para que sus hijos puedan seguir aprendiendo. Realmente es algo simbólico, cada libro costará solo USD*NUMBER* pesitos, es un precio muy bajo si tenemos en cuenta que con esto los niños podrán forjarse un futuro. Decidimos contratar a Rosado porque queríamos hacer libros de calidad, que fueran atractivos para las nueva generaciones", dijo,\nNuño menciona que el conductor explicará a los estudiantes las materias de matemáticas, historia, geografía, español", y otros temas, en un idioma más alegre y jocoso para hacer más disfrutable el aprendizaje:\n"Queremos que los niños lleguen a sus casas y abran gustosos sus libros de texto en vez de que se aburran al hojearlos. ¿Qué ondiux con las matemáticas?, será el ejemplar de matemáticas, ¿Qué pex con el spanish?, será la edición para español. Cada texto será redactado de forma atractiva y tendrá padrísimos dibujos e ilustraciones diseñadas por el mismo Rosado", comentó.\nEl director menciona que Yordi redactará los libros de primaria, secundaria y prepa: "él domina muy bien el lenguaje de los adolescentes, pre adolescentes y niños. Realmente hizo una labor de investigación muy grande para entender y dominar a la perfección el lenguaje de las nuevas generaciones, los estudiantes definitivamente van a disfrutar los textos y se van a identificar con el idioma juvenil. Rosado respetó los contenidos sin alterarlos pero logró convertirlos al lenguaje de la chaviza, hasta yo que ya estoy grande me quedé picado al leerlos, la neta están padrísimos", puntualizó riendo el director. YORDI ROSADO ESCRIBIRÁ Y DISEÑARÁ LOS NUEVOS LIBROS DE TEXTO DE LA SEP PARA HACERLOS MÁS ATRACTIVOS', '1'
#Ejemplo: 'UNAM capacitará a maestros para aprobar prueba Pisa\nLa máxima casa de estudios y la SEP firmaron cinco convenios para que las facultades de Ciencias y Química, así como el Instituto de Matemáticas de enseñen a los profesores estrategias para impartir estas disciplinas a los alumnos de preescolar, primaria, secundaria\nLa Universidad Nacional Autónoma de México (UNAM) capacitará a profesores de educación obligatoria en matemáticas, ciencias y lectura en la enseñanza de estas materias en las escuelas públicas ante la necesidad de mejorar los resultados de México en las pruebas internacionales de aprovechamiento escolar, como PISA.\n Durante la firma de un convenio general y cuatro específicos de colaboración entre la Universidad Nacional y la Secretaría de Educación Pública (SEP), el rector Enrique Graue Wiechers señaló que uno de los objetivos de estos instrumentos será que las facultades de Ciencias y Química, así como el Instituto de Matemáticas de la máxima casa de estudios colaboren con la dependencia federal para ofrecer cursos de capacitación a los maestros mexicanos. UNAM capacitará a maestros para aprobar prueba Pisa', '0'
#Ejemplo: 'UNAM REALIZARÁ PRUEBAS ANTIDOPING A ESTUDIANTES DE SOCIOLOGÍA Y FILOSOFÍA; EXPULSARÁN A LOS QUE DEN POSITIVO\nMéxico.- El rector de la Universidad Autónoma de México, Enrique Graue Wiechers, reveló en conferencia de prensa que a partir del próximo ciclo escolar, estudiantes de las carreras de Sociología y Filosofía serán sometidos a un examen antidoping que se les aplicará cada *NUMBER* meses, y los alumnos que den positivo serán expulsados definitivamente de la institución, esto con la finalidad de frenar la venta y consumo de drogas dentro de las instalaciones, además de limpiar la fama que dichas licenciaturas tienen de ser "para puro marihuano y borracho".\nEnrique dejo en claro que es bien sabido por todos que en dichas facultades es donde más se consumen sustancias ilícitas dentro de las instalaciones, ya que incluso los alumnos se han apoderado de ciertas áreas para irse a drogar en horarios de clase:\n"Hemos sido muy tolerantes con los alumnos en general: cuando descubrimos a un alumno (independientemente de la facultad en la que se encuentre) consumiendo bebidas alcohólicas o cometiendo una falta al reglamento, inmediatamente es trasladado a rectoría para imponerle un castigo (que incluso puede llegar a la expulsión), sin embargo los vigilantes y cuerpo de seguridad de Sociología y Filosofía prefieren hacerse de la vista gorda porque en dichas carreras ya es algo común que los alumnos todo el día se estén drogando, y es imposible controlarlos a todos.\nLos anteriores rectores les dieron muchas libertades a estos estudiantes pero ya es hora de ponerles un alto, una cosa es que quieran luchar contra el sistema, y otra romper las reglas y luego hacerse los ofendidos", comentó.\nGraue recordó el caso de "El Yorch", un estudiante que hace meses fue detenido por traficar droga dentro en las inmediaciones del auditorio de Filosofía:\n"Ese es el caso más conocido pero no es el único del que se tiene registro, la realidad es que decenas de alumnos de estas carreras venden, compran, trafican y consumen drogas dentro de las instalaciones como si nada.\nLa venta y consumo se da en todas las carreras no lo voy a negar, pero en Filosofía y Sociología lo hacen de un modo descarado. Es un hecho que realizaremos las pruebas antidoping y al que no le guste que se salga, algunos ya amenazaron con ir a Derechos Humanos pero nosotros tenemos la libertad de implementar cualquier regla siempre y cuando no violentemos física ni psicológicamente a alguien.\nLos alumnos que opten por seguir drogándose serán obligados a dejar la institución para cederle su lugar a alguien que realmente tenga deseos de ir a estudiar, y no a agarrar a la facultad como cantina o bar para echarse sus churros de mota o tomarse sus aguas locas", puntualizó. UNAM REALIZARÁ PRUEBAS ANTIDOPING A ESTUDIANTES DE SOCIOLOGÍA Y FILOSOFÍA; EXPULSARÁN A LOS QUE DEN POSITIVO', '1'
#Ejemplo: 'Niño de *NUMBER* años se prepara para entrar a la universidad\nCon un coeficiente intelectual de *NUMBER*, Laurent Simons terminó su educación secundaria en año y medio; tiene dificultad para relacionarse con otros niños; sigue indeciso sobre su futuro\nAunque la mayoría de los niños en algún punto dicen \'odiar la escuela y no querer volver a ir\' existe un pequeño para el que los estudios lo son todo.\nLaurent Simons es un niño de ocho años nacido en Brujas, Bélgica, que ha acaparado la atención de los medios por estar listo para entrar a la universidad tras terminar con un promedio perfecto su educación secundaria en tan sólo *NUMBER* meses.\nTambién te recomendamos leer: Niño pidió ser una concha en su fiesta, ¡lo más tierno que verás hoy!\nDe acuerdo con información del canal belga VRT, comenzó su educación secundaria en *NUMBER* en Ámsterdan y la culminó en tiempo récord gracias a su coeficiente intelectual de *NUMBER*, muy superior al rango de entre *NUMBER* y *NUMBER* asociado a la inteligencia media. A su corta edad habla francés, alemán y holandés.\nAunque su inteligencia le ha abierto muchas puertas también ha sido obstáculo para ciertas cosas, como por ejemplo, hacer amigos.\nSegún sus padres, Simons prefiere estudiar antes que jugar con otros niños.\nHasta hace poco, el pequeño tenía en mente convertirse en cirujano o astronauta, pero ahora está más interesado en la ingeniería porque le gustan las matemáticas.\nSon muy vastas y tienen muchas ramas", dice.\nAfortunadamente, esta decisión no es algo que les preocupe a sus progenitores, pues lo único que desean para su hijo es que sea feliz. Si decidiera ser carpintero, no sería ningún problema. Lo importante es que sea feliz", afirmó su padre Alexander. Niño de *NUMBER* años se prepara para entrar a la universidad', '0'
#Ejemplo: 'LIMITARÁN EL TIEMPO DE EGRESO EN FILOSOFÍA Y SOCIOLOGÍA A MÁXIMO *NUMBER* AÑOS PARA EVITAR LA FORMACIÓN DE "FÓSILES"\nMéxico.- El rector de la UNAM, Enrique Luis Graue, citó a los medios a una conferencia de prensa para informar que a partir del próximo año el tiempo de egreso de las carreras de Filosofía y Sociología de las Universidades de todo el país será limitado a un máximo de "*NUMBER* años" (solo uno más de lo que normalmente dura la licenciatura) con la finalidad de frenar la formación de "fósiles" y así evitar que las universidades tengan un gasto innecesario.\n Enrique señala que decidieron implementar la estrategia en dichas carreras ya que es donde se registra mayor número de "fósiles", es decir, alumnos que reprueban materias de adrede, toman solo una o dos por año, o se salen de la escuela para regresar el próximo semestre con la finalidad de prolongar su estadía en la Universidad ya que que se convierte en su zona de confort:\nTe puede interesar Lolita de la Vega gana demanda para evitar que TV Azteca vuelva a retrasmitir Sailor Moon\n "Las licenciaturas de Filosofía y Sociología están diseñada para acabarse en un máximo *NUMBER* años, en estas carreras es donde se registra un mayor número de chairos guerrilleros pero también en donde existe el mayor número de fósiles. Para estas personas su institución se convierte en su zona de confort y hacen lo posible por prolongar su estadía porque sienten un miedo terrible de salir al mundo real. En la Universidad se sienten parte de algo, tienen miedo que al egresar se conviertan en fantasmas y nadie los tome en cuenta. Cada alumno de una Universidad pública le cuesta más de USD*NUMBER* mil pesos semestrales al estado, no es justo que el gobierno pague dicha cantidad por alguien que solo se inscribe para tomar una o dos materias cada *NUMBER* meses para alargar su estadía, que muchas veces llega a ser hasta de *NUMBER* años", dijo.\n Luis mencionó que estas licenciaturas duran *NUMBER* semestres, y a partir del próximo año solo se les dará oportunidad de estudiar en la institución otros *NUMBER* semestres más en caso de que se retrasen: "hay gente que literalmente dura *NUMBER* semestres en la Universidad, esto se va a terminar. Los alumnos de estas carreras no podrán dar de bajas materias, y si reprueban de adrede los pasaremos aunque no quieran. Si aun así el estudiante se empeña en reprobar para no salir y desaprovecha su año de tolerancia entonces será dado de baja de manera automática y no podrá volver a ingresar a la Universidad nunca más, ni a su carrera ni a ninguna otra. No es justo que su capricho le provoque al gobierno un gasto millonario incensario que puede ser invertido en infraestructura o nuevo material. A partir del *NUMBER* tendrán un año de tolerancia para egresar, es la única manera en que podemos acabar con los fósiles", puntualizó. LIMITARÁN EL TIEMPO DE EGRESO EN FILOSOFÍA Y SOCIOLOGÍA A MÁXIMO *NUMBER* AÑOS PARA EVITAR LA FORMACIÓN DE "FÓSILES"', '1'
#Ejemplo: 'La Universidad de Oxford da más tiempo a las mujeres para hacer los exámenes\nLa polémica medida que ha puesto en marcha la Universidad se da para intentar mejorar los resultados de las estudiantes.\nLa Universidad de Oxford ha puesto en marcha una medida para buscar la mejora de las notas en sus estudiantes. A pesar de que la medida comenzó a implantarse hace unos meses, concretamente en junio del *NUMBER*, ha trascendido ahora gracias a las informaciones ofrecidas por el medio británico The Telegraph.\nLa medida ha generado cierta polémica debido a su carácter desigual con los sexos de sus estudiantes, ya que la Universidad ha decidido dar *NUMBER* minutos más de tiempo a las mujeres para realizar un examen respecto a los estudiantes varones. Este aumento de duración de los exámenes viene motivado por el bajo rendimiento de las mujeres en las notas de los exámenes respecto al de los hombres.\nEn anteriores años las notas altas en los hombres duplicaban a las de las mujeres y con este cambio se pretende ecualizar estos desajustes. La Universidad indica que este aumento de duración con el que las mujeres disponen de *NUMBER* minutos respecto a los habituales *NUMBER* minutos se hace con motivos reales en los que las mujeres "manejan peor el estrés" y "repasan más sus respuestas".\nNo obstante, y tras la polémica generada en tan curiosa medida que está realizada con el fin de "mitigar la brecha de género que ha aparecido en los últimos años", la Universidad no piensa dar marcha atrás con la misma. La razón es que los resultados de esta medida han sido satisfactorios pues se ha conseguido mejorar la nota media de las estudiantes.\nMuchas voces críticas respecto a la situación\nLa medida ha causado gran polémica pues muchos expertos han asegurado que no hay diferencias de género en lo referente a las habilidades matemáticas. Además de ello, a pesar de que la Universidad indica que lo hacen para igualar géneros, la percepción es la de causar una gran desigualdad que hace ver que las mujeres siguen siendo el "sexo débil".\nLas redes por su lado han mostrado su incredulidad al respecto y critican la medida en la que sienten que se genera una minusvaloración de las mujeres respecto al hombre. La Universidad de Oxford da más tiempo a las mujeres para hacer los exámenes'. '1'
#Ejemplo: 'Escultura de un angel caído genera un fuerte impacto en las redes sociales\n Impresionante repercusión por la obra de un angel caído hiperrealista en China\n Los artistas Sun Yuan y Peng Yu presentaron en la capital de China una escultura de una anciana con las alas desplumadas y tirada en el suelo que causó una fuerte impresión y dio la vuelta al mundo\nLa obra hiperrealista de un ángel caído causó un fuerte impacto en China, y se viralizó a través de Internet a todo el mundo. Los artistas de la obra en cuestión se llaman Sun Yuan y Peng Yu, y la noticia se difundió a través del diario People de China .\nLos artistas son originarios de Beijing y son muy conocidos por trabajar con materiales hiperrealistas como grasa humana. Su último trabajo, "Ángel", que muestra a una anciana tirada en el suelo con unas alas desplumadas, generó un fuerte impacto en los espectadores.\nEsta obra fue realizada con gel siliconado, fibra de vidrio y acero inoxidable, entre otros materiales. Los artistas explicaron al medio chino que su intención era ilustrar la tensión y los cambios entre el surrealismo y el realismo. El ángel que tenía poderes terminó sin servir para nada, y mucho menos para ayudar a otros a cumplir la voluntad de Dios.\nEn la página web de los artistas agregaron que Ángel es una escultura en tamaño real que tiene que ver con su nuevo enfoque: "El ángel, una mujer anciana en un camisón blanco con sus alas desplumadas, está tirada boca abajo en el piso; quizás durmiendo, quizás muerta, pero ciertamente inmóvil y congelada en una imagen demasiado realista".\n"El ser sobrenatural, ahora nada más que una impotente criatura, no puede ni llevar adelante su voluntad, ni ayudar a quienes creen en su existencia. El ángel es real pero ineficiente; los sueños y la esperanza son reales pero vanos", finalizaron. Escultura de un angel caído genera un fuerte impacto en las redes sociales', '0'
###Input: Este es la noticia: {}

### Response:
"""



for index, row in df.iterrows():
  messages = [
      {"role": "system", "content": "Eres un experto en Ciencias de la Comunicación Política. Se te ha proporcionado una instrucción que describe una tarea y se combina con una entrada que brinda más contexto. Responde según lo indicado en la instrucción."},
      {"role": "user", "content": prompt.format(row['Text'])}
  ]
  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )
  model_inputs = tokenizer([text], return_tensors="pt").to(device)

  # Directly use generate() and tokenizer.decode() to get the output.
  # Use `max_new_tokens` to control the maximum output length.
  generated_ids = model.generate(
      model_inputs.input_ids,
      max_new_tokens=1,
      temperature = temp
  )
  generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]

  sys_ans.append(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

print(sys_ans)

sys_ans = [1 if item == '1' else 0 for item in sys_ans]

gold_ans = df['Category'].tolist()




print("gold_ans:", gold_ans)
print("list_answer:", sys_ans)





def compute_metrics(predictions, labels):
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")


        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
            "accuracy"
        ]
        precision = precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["precision"]
        recall = recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["recall"]
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"]

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }


results=compute_metrics(sys_ans, gold_ans)
print(prompt)
print(f"Allsides, {configuration}, prompt with temperature={temp},  n_labels=ALL:\n", json.dumps(results, indent=4))
# Compute precision, recall, and f1-score
print(f"Allsides, {configuration}, prompt with temperazture={temp}, n_labels=ALL:\n", classification_report(gold_ans,sys_ans))

print("My program took", time.time() - start_time, "to run")
