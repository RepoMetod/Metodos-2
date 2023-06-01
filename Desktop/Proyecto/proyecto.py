import streamlit as st
import pandas as pn

def main():
    #titulo
    st.title("Metodos Numericos II")
    #titulo sidebar
    st.sidebar.header("Apuntes")
    # Primera sección
    st.sidebar.subheader("Unidad 1")
    metodo = st.sidebar.selectbox("Metodos", [" ","Sistema de ecuaciones", "Polinomio de Taylor", "Polinomio de Taylor Multivariable","Función Inversa","Newton Raphson","Newton Multivariable","Quasi-Newton","Raices de Broyden","BFGS"])
    #segunda seccion
    st.sidebar.header("Unidad 2")
    metodos2 = st.sidebar.selectbox("Metodos", [" ","Lagrange", "Splines Cubicos", "Hermite","OLS"]) 
    #Tercera seccion
    st.sidebar.header("Unidad 3")
    metodos3 = st.sidebar.selectbox("Metodos", [" " ,"Minimos Cuadrados", "Descenso Gradiente", "Batch y MiniBatch","Regla Trapezoidal","Regla Simpson 1/3","Regla Simpson 3/8"]) 

    # Primera sección Seleccionadores
    st.sidebar.subheader("Complementacion")
    if metodo == "Sistema de ecuaciones":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo='''
            import numpy as np

            # Definición del sistema de ecuaciones
            # Ax = b
            A = np.array([[2, 3, -1],
                            [4, 1, 2],
                            [-1, 2, 3]])

            b = np.array([4, 1, 3])

            # Resolución del sistema de ecuaciones
            x = np.linalg.solve(A, b)

            # Imprimir la solución
            st.write("La solución del sistema de ecuaciones es:")
            st.write(x)'''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Dowlander
            import numpy as np
            # Definición del sistema de ecuaciones
            # Ax = b
            A = np.array([[2, 3, -1],
                        [4, 1, 2],
                        [-1, 2, 3]])

            b = np.array([4, 1, 3])

            # Resolución del sistema de ecuaciones
            x = np.linalg.solve(A, b)
            # Mostrar la solución en la interfaz
            st.write("La solución del sistema de ecuaciones es:")
            st.write(x)

        if st.sidebar.checkbox("Ejemplo"):
            st.title("**Ejemplo:**")
            texto = "<p style='text-align:justify; text-justify: inter-word;'>Tenermos un sistema de ecuaciones lineales de tres incognitas (x,y,z) y tres ecuaciones lineales.</p>"
            texto += "<p style ='text-align:justify;text-justify: inter-word;'>El objetivo es encontrar los valores de x,y,z que satisfacen simultaneamente las tres ecuaciones.</p>"
            st.markdown(texto,unsafe_allow_html=True)
            st.latex(r'x+2y-z = 0 \\ 3x-y+4z = 0 \\ 2x+3y+z=0')
            st.write("Para solucionar el sistema de ecuaciones, primero escribimos la matriz ampliada del sistema de la forma:")
            st.latex(r'\begin{bmatrix} 2 & 1 & -1 & 2 \\ 4 & 5 & -3 & 6 \\ 2 & 10 & 4 & 16 \end{bmatrix}')
            st.write("Aplicamos las operaciones elementales para convertir la matriz ampliada a una escalonada reducida.")
            st.latex(r'\begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 2 \end{bmatrix}')
            st.write("Por lo tanto, la solución del sistema de ecuaciones es x=1 , y=1 y z=2.")

    elif metodo == "Polinomio de Taylor":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo='''
            import sympy as sp

            # Definición de la variable simbólica x
            x = sp.Symbol('x')

            # Definición de la función f(x)
            f = sp.sin(x)

            # Cálculo del polinomio de Taylor de f(x) alrededor de x0 = 0
            polinomio_taylor = f.series(x, 0, 5).removeO()

            # Imprimir el polinomio de Taylor
            print("El polinomio de Taylor de f(x) alrededor de x = 0 es:")
            print(polinomio_taylor)'''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="polinomio_taylor.py",
                mime="text/palin")

            #Codigo funcional
            import sympy as sp

            # Definición de la variable simbólica x
            x = sp.Symbol('x')

            # Definición de la función f(x)
            f = sp.sin(x)

            # Cálculo del polinomio de Taylor de f(x) alrededor de x0 = 0
            polinomio_taylor = f.series(x, 0, 5).removeO()

            # Imprimir el polinomio de Taylor
            st.write("El polinomio de Taylor de f(x) alrededor de x = 0 es:")
            st.write(polinomio_taylor)

        if st.sidebar.checkbox("Ejemplo"):
            st.title("**Ejemplo**")
            texto = "<p style='text-aling:justify;text-justify:inter-word;'>El polinomio de taylor es una aproximacion de una funcion alrededor de un punto.</p>"
            texto+="<p style='text-aling:justify;text-justify:inter-word;'>Queremos aproximar la función seno alredeor de un punto x=0, el polinomio de grado n se puede escribir como: </p>"
            st.markdown(texto,unsafe_allow_html=True)
            st.latex(r'P_n(x)=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\frac{x^7}{7!}+...+\frac{(-1)^n x^{2n+1}}{(2n+1)!}')
            texto+="<p style='text-aling:justify;text-justify:inter-word;'>Queremos aproximarlo a una funcion seno,por ejemplo, lo queremos aproximar a sen(0.3), calculamos p5(0.3): </p>"
            st.markdown(texto,unsafe_allow_html=True)
            st.latex(r'P5(0.3) = 0.3-^\frac{0.3^3}{3!}+\frac{0.3^5}{5!}-\frac{0.3^7}{7!}+\frac{0.3^9}{9!} \approx 0.2955.')
            st.write("Este valor es una buena aproximacion de sen(0.3), ya que el gradoe es 5 y es cercano al punto x=0.")
    elif metodo == "Polinomio de Taylor Multivariable":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo= '''
            import sympy as sp
            # Definición de las variables simbólicas x, y
            x, y = sp.symbols('x y')
            # Definición de la función f(x, y)
            f = x**2 + y**3
            # Punto de expansión
            x0 = 1
            y0 = 2
            # Orden del polinomio de Taylor
            orden = 2
            # Cálculo del polinomio de Taylor de f(x, y) alrededor de (x0, y0)
            polinomio_taylor = f.series(x, x0, orden).series(y, y0, orden)
            # Imprimir el polinomio de Taylor
            print("El polinomio de Taylor de f(x, y) alrededor de (x0, y0) es:")
            print(polinomio_taylor)'''
            st.code(codigo,language='python')
            #Downlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="polinomio_taylor_Multi.py",
                mime="text/palin")
            #Codigo funcional
            import sympy as sp

            # Definición de las variables simbólicas x, y
            x, y = sp.symbols('x y')

            # Definición de la función f(x, y)
            f = x**2 + y**3

            # Punto de expansión
            x0 = 1
            y0 = 2

            # Orden del polinomio de Taylor
            orden = 2

            # Cálculo del polinomio de Taylor de f(x, y) alrededor de (x0, y0)
            polinomio_taylor = f.series(x, x0, orden).removeO().series(y, y0, orden).removeO()

            # Imprimir el polinomio de Taylor
            st.write("El polinomio de Taylor de f(x, y) alrededor de (x0, y0) es:")
            st.write(polinomio_taylor)


        if st.sidebar.checkbox("Ejemplo"):
            st.title("**Ejemplo:**")
            st.write("el polinomio de Taylor multivariable de la función:")
            st.latex(r'f(x,y)=x^2y+2xy+y^2' "entorno al punto (1,1) es:")
            st.latex(r'f(x,y)\approx f(1,1)+\frac{\partial f}{\partial x (1,1)(x-1)}+\frac{\partial f}{\partial y(1,1)(y-1)}+\frac{1}{2!}(\partial ^2f / \partial x^2 (1,1))+\frac{\partial ^2f}{\partial^2 y}(1,1)(y-1)^2')
            st.write("Calculamos las derivadas parciales y las segundas derivadas parciales: ")
            st.latex(r'\frac{\partial f}{\partial x}= 2xy + 2y')
            st.latex(r'\frac{\partial f}{\partial y}=x^2+2x+2u')
            st.latex(r'\frac{\partial ^2f}{\partial ^2x}=2y')
            st.latex(r'\frac{\partial ^2f}{\partial ^2y}=2')
            st.latex(r'\frac{\partial ^2f}{\partial x \partial y}=2x+2')
            st.write("Evaluamos en el punto (1,1), obteniendo: ")
            st.latex(r'f(1,1)=4')
            st.latex(r'\frac{\partial f}{\partial x}= 4')
            st.latex(r'\frac{\partial f}{\partial y}=5')
            st.latex(r'\frac{\partial ^2f}{\partial ^2x}=2')
            st.latex(r'\frac{\partial ^2f}{\partial ^2y}=2')
            st.latex(r'\frac{\partial ^2f}{\partial x \partial y}=4')
            st.write("Sustituyendo estos valores en la formula del polinomio de taylor multivariable, se obtinen:")
            st.latex(r'f(x,y)\approx 4+4(x-1)+5(y-1)+(1/2!)(2(x-1)^2+2(x-1)(y-1)+2(y-1)^2)')
            st.latex(r'f(x,y\approx 2x^2+5xy-5x+y^2+3y-1')
    elif metodo == "Función Inversa":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo='''
            import sympy as sp

            # Definición de las variables simbólicas x, y
            x, y = sp.symbols('x y')

            # Definición de la función f(x) = x**3 - x + 2
            f = x**3 - x + 2

            # Derivada de f con respecto a x
            df_dx = sp.diff(f, x)

            # Comprobación de la condición de existencia de la inversa
            if df_dx.subs(x, 1) != 0:
                print("La función f(x) cumple la condición de existencia de la inversa en x = 1.")
            else:
                print("La función f(x) no cumple la condición de existencia de la inversa en x = 1.")
            '''
            st.code(codigo,language='python')
            #Downlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="Función_Inversa.py",
                mime="text/palin")
            #Codigo funcional
            import sympy as sp

            # Definición de las variables simbólicas x, y
            x, y = sp.symbols('x y')

            # Definición de la función f(x) = x**3 - x + 2
            f = x**3 - x + 2

            # Derivada de f con respecto a x
            df_dx = sp.diff(f, x)

            # Comprobación de la condición de existencia de la inversa
            if df_dx.subs(x, 1) != 0:
                st.write("La función f(x) cumple la condición de existencia de la inversa en x = 1.")
            else:
                st.write("La función f(x) no cumple la condición de existencia de la inversa en x = 1.")



        if st.sidebar.checkbox("Ejemplo"):
            st.title("***Ejemplo***")
            st.write("Entonces tenemos que resolver lo siguiente:")
            st.latex(r'f(x)=x^3')
            st.write("dertemina") 
            st.latex(r'(F^{-1}´(8))')
            st.write("Tenemos que:")
            st.latex(r'(f^{-1})´(8) = \frac{1}{f´x_0}')
            st.latex(r'f´(x)=3x^2')
            st.write("Pero tambien necesitamos saber quien es x0. Para ello tenemos que hacer:")
            st.latex(r'8=f(x_0)')
            st.latex(r'8=(x_0)^3=x_0=3\sqrt{8}=x_0=2')
            st.write("Aplicamso el teorema:")
            st.latex(r'(f^{-1})´(8)=\frac{1}{3x2^2}=\frac{1}{12}')
            st.write("Si calculamos la inversa de f y derivamos y aplicamos el teorema, encontramos exactamente el mismo resultado.")
    elif metodo == "Newton Raphson":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt

            def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
                x = x0
                iter_count = 0
                while True:
                    # Evaluación de la función y su derivada en el punto actual
                    f_val = f(x)
                    df_val = df(x)
                    # Cálculo del cambio en x
                    dx = -f_val / df_val
                    # Actualización del punto
                    x += dx
                    iter_count += 1
                    # Comprobación de la condición de convergencia
                    if abs(dx) < tol or iter_count >= max_iter:
                        break
                return x

            # Ejemplo de uso
            def f(x):
                return x**2 - 4

            def df(x):
                return 2*x

            x0 = 1.5
            result = newton_raphson(f, df, x0)
            st.write("Raíz encontrada:", result)

            # Crear un rango de valores x
            x_values = np.linspace(-5, 5, 100)
            # Evaluar la función f(x) en los valores x
            y_values = f(x_values)

           # Configurar la gráfica
            fig, ax = plt.subplots()
            ax.plot(x_values, y_values, label='f(x) = x^2 - 4')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.scatter(result, f(result), color='red', label='Raíz encontrada')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfico de la función f(x)')
            ax.legend()
            ax.grid(True)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo, language='python')
            #Downlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="Newton-Raphson.py",
                mime="text/palin")
            #Codigo funcional
            import numpy as np
            import matplotlib.pyplot as plt

            def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
                x = x0
                iter_count = 0
                while True:
                    # Evaluación de la función y su derivada en el punto actual
                    f_val = f(x)
                    df_val = df(x)
                    # Cálculo del cambio en x
                    dx = -f_val / df_val
                    # Actualización del punto
                    x += dx
                    iter_count += 1
                    # Comprobación de la condición de convergencia
                    if abs(dx) < tol or iter_count >= max_iter:
                        break
                return x

            # Ejemplo de uso
            def f(x):
                return x**2 - 4

            def df(x):
                return 2*x

            x0 = 1.5
            result = newton_raphson(f, df, x0)
            st.write("Raíz encontrada:", result)

            # Crear un rango de valores x
            x_values = np.linspace(-5, 5, 100)
            # Evaluar la función f(x) en los valores x
            y_values = f(x_values)

            # Configurar la gráfica
            fig, ax = plt.subplots()
            ax.plot(x_values, y_values, label='f(x) = x^2 - 4')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.scatter(result, f(result), color='red', label='Raíz encontrada')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Gráfico de la función f(x)')
            ax.legend()
            ax.grid(True)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)


        if st.sidebar.checkbox("Ejemplo"):
            st.title("**Ejemplo:**")
            st.write("Usando el metodo Newton-Raphson para encontrar una raiz negativa del siguiente polinomio cúbico.")
            st.latex(r'x^3-2x+1')
            st.write("Iniciamos con el punto de partida con el valor x0=-1.5 para encontrar aproximacionesucesivas con tres decimales de precisión.")
            st.write("Entonces, Definimos la funcion a la cual se le hallaran los ceros o raíces.")
            st.latex(r'f(x)=x^2-2x+1')
            st.write("Hallamos la derivada de la función:")
            st.latex(r'f´(x)=3x^2-2')
            st.write("Elegimos el punto de partida, en este ejemplo el valor sera de x0=-1.5.")
            st.write("Evaluamos la funcion y su derivada en x0:")
            st.latex(r'f(x_0)=0.625 y f´(x_0)=4.75')
            st.write("Aplicamos la formula iterativa de Newton-Raphson para hallar una primera estimacion:")
            st.latex(r'x_1 = x_0 - \frac{f(x_0)}{f´(x_0)}')
            st.latex(r'x_1 = -1.6316')
            st.write("Repetimos la evaluacion de la funcion y derivada de (xi+1) y la aplicacion de la formula hasta que la estimacion coincida a los decimales deceados:")
            st.latex(r'f(x_1) = -8.0187 x 10^{-1} f´(x_1)=5.9861')
            st.latex(r'x_2=x_1 - \frac{f(x_1)}{f´(x_1)}  x_2=-1.6182')
            st.latex(r'f(x_2)=-8.7589 x 10^{-4}  f´(x_2)=5.8556')
            st.latex(r'x_3=x_2-\frac{f(x_2)}{f´(x_2)}  x_3=-1.618')
            st.write("Dado que se han repetido las tres cifras decimales entre la segunda y tercera iteracion, solo fue necesario tomar 3 iteraciones.")
            st.write("Se toma como valor inicial de la raiz el valor de la ultima iteracion en este caso, x=-1.618.")
    elif metodo == "Newton Multivariable":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo='''
            import matplotlib.pyplot as plt

            def newton_multivariable(f, x0, tol=1e-6, max_iter=100):
                x = x0
                iter_count = 0
                while True:
                    # Evaluación de la función y su Jacobiano en el punto actual
                    f_val = f(x)
                    J = jacobian(f, x)
                    # Resolución del sistema de ecuaciones lineales J*dx = -f(x)
                    dx = np.linalg.solve(J, -f_val)
                    # Actualización del punto
                    x += dx
                    iter_count += 1
                    # Comprobación de la condición de convergencia
                    if np.linalg.norm(dx) < tol or iter_count >= max_iter:
                        break
                return x

            def jacobian(f, x):
                n = len(x)
                J = np.zeros((n, n))
                # Cálculo numérico de las derivadas parciales
                for i in range(n):
                    h = 1e-6
                    x_plus_h = x.copy()
                    x_plus_h[i] += h

                    J[:, i] = (f(x_plus_h) - f(x)) / h
                return J

            # Ejemplo de uso
            def f(x):
                return np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])

            x0 = np.array([0.5, 0.5])
            result = newton_multivariable(f, x0)
            st.write("Solución:", result)

            # Crear un rango de valores x y y
            x_values = np.linspace(-2, 2, 100)
            y_values = np.linspace(-2, 2, 100)
            X, Y = np.meshgrid(x_values, y_values)

            # Evaluar la función f(x, y) en los valores x y y
            Z1 = X**2 + Y**2 - 1
            Z2 = X - Y

           # Configurar la gráfica
            fig, ax = plt.subplots()
            ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
            ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
            ax.scatter(result[0], result[1], color='green', label='Solución')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Gráfico de las ecuaciones f(x, y)')
            ax.legend()
            ax.grid(True)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Downlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="Newton-Multivariado.py",
                mime="text/palin")
            #Codigo Funcional
            import numpy as np
            import matplotlib.pyplot as plt

            def newton_multivariable(f, x0, tol=1e-6, max_iter=100):
                x = x0
                iter_count = 0
                while True:
                    # Evaluación de la función y su Jacobiano en el punto actual
                    f_val = f(x)
                    J = jacobian(f, x)
                    # Resolución del sistema de ecuaciones lineales J*dx = -f(x)
                    dx = np.linalg.solve(J, -f_val)
                    # Actualización del punto
                    x += dx
                    iter_count += 1
                    # Comprobación de la condición de convergencia
                    if np.linalg.norm(dx) < tol or iter_count >= max_iter:
                        break
                return x

            def jacobian(f, x):
                n = len(x)
                J = np.zeros((n, n))
                # Cálculo numérico de las derivadas parciales
                for i in range(n):
                    h = 1e-6
                    x_plus_h = x.copy()
                    x_plus_h[i] += h

                    J[:, i] = (f(x_plus_h) - f(x)) / h
                return J

            # Ejemplo de uso
            def f(x):
                return np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])

            x0 = np.array([0.5, 0.5])
            result = newton_multivariable(f, x0)
            st.write("Solución:", result)

            # Crear un rango de valores x y y
            x_values = np.linspace(-2, 2, 100)
            y_values = np.linspace(-2, 2, 100)
            X, Y = np.meshgrid(x_values, y_values)

            # Evaluar la función f(x, y) en los valores x y y
            Z1 = X**2 + Y**2 - 1
            Z2 = X - Y

            # Configurar la gráfica
            fig, ax = plt.subplots()
            ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
            ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
            ax.scatter(result[0], result[1], color='green', label='Solución')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Gráfico de las ecuaciones f(x, y)')
            ax.legend()
            ax.grid(True)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)



        if st.sidebar.checkbox("Ejemplo"):
            st.title("***Ejemplo***")
            st.write("Consideremos la funcion ") 
            st.latex(r'f(x,y)=x^2+y^2') 
            st.write("Queremos encontrar el minimo de esta funcion.")
            st.write("Entonces:")
            st.write("Inicializamos eligiendo un punto inicial en (x0,y0) en el dominio de la funcion.Por ejemplo (x0,y0)=(1,1).")
            st.write("Iniciamos Iteraciones.")
            st.write("Calculamos el gradiente de la funcion f(x,y):")
            st.latex(r'\nabla f(x,y) = (2x,2y)')
            st.write("Calculamos la matriz Hessiana de la funcion f(x,y)")
            st.latex(r'Hf(x,y) = |2 0|')
            st.write("Calculamos la direccion de busqueda utilizando la formula:")
            st.latex(r'd = -{Hf(x,y)^{-1} * \nabla f(x,y)}')
            st.write("Actualizando el punto actual (x0,y0) utilizando la formula:")
            st.write("Repetimos los pasos anteriores hasta que se alcance la convergencia deseada o se cumpla un criterio de parada.")
            st.write("En cada iteracion, se calcula la direccion de busqueda basada en la matriz Hessiana y el gradiente, y se actualiza el punto actual utilizado en la direccion.")
            st.write("En el caso de la funcion el metodo convergera rapidamente al minimo global en (0,0)")
    elif metodo == "Quasi-Newton":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt

            def bfgs(f, df, x0, tol=1e-6, max_iter=100):
                n = len(x0)
                I = np.eye(n)  # Matriz identidad
                H = I  # Inicialización de la matriz H
                x = x0
                iter_count = 0
                path = [x]  # Lista para almacenar el camino de optimización

                while True:
                    f_val = f(x)
                    df_val = df(x)

                    p = -np.dot(H, df_val)  # Dirección de búsqueda

                    alpha = line_search(f, df, x, p)  # Longitud de paso mediante búsqueda lineal

                    x_new = x + alpha * p  # Nuevo punto

                    s = x_new - x
                    y = df(x_new) - df_val

                    rho = 1 / np.dot(y, s)

                    H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

                    x = x_new
                    iter_count += 1
                    path.append(x)

                    if np.linalg.norm(alpha * p) < tol or iter_count >= max_iter:
                        break

                return x, path

            def line_search(f, df, x, p):
                alpha = 1.0
                c = 0.9  # Factor de reducción del paso
                rho = 0.5  # Factor de reducción del tamaño de búsqueda

                while f(x + alpha * p) > f(x) + c * alpha * np.dot(df(x), p):
                    alpha *= rho

                return alpha

            # Ejemplo de uso
            def f(x):
                return x[0]**2 + x[1]**2

            def df(x):
                return np.array([2*x[0], 2*x[1]])

            x0 = np.array([1.0, 1.0])
            solution, path = bfgs(f, df, x0)

            # Crear una malla de puntos para la visualización del gráfico
            x_values = np.linspace(-2, 2, 100)
            y_values = np.linspace(-2, 2, 100)
            X, Y = np.meshgrid(x_values, y_values)
            Z = X**2 + Y**2  # Función objetivo

            # Graficar la función objetivo
            plt.contour(X, Y, Z, levels=20, cmap='viridis')

            # Graficar el camino de optimización
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'ro-')

            # Graficar la solución encontrada
            plt.plot(solution[0], solution[1], 'go', label='Solución')

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Optimización con Quasi-Newton (BFGS)')
            plt.legend()
            plt.grid(True)

            # Mostrar el gráfico en Streamlit
            st.pyplot()
            '''
            st.code(codigo,language='python')
            #Downlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="Quasi-Newton.py",
                mime="text/palin")
            #Codigo funcional
            import numpy as np
            import matplotlib.pyplot as plt

            def bfgs(f, df, x0, tol=1e-6, max_iter=100):
                n = len(x0)
                I = np.eye(n)  # Matriz identidad
                H = I  # Inicialización de la matriz H
                x = x0
                iter_count = 0
                path = [x]  # Lista para almacenar el camino de optimización

                while True:
                    f_val = f(x)
                    df_val = df(x)

                    p = -np.dot(H, df_val)  # Dirección de búsqueda

                    alpha = line_search(f, df, x, p)  # Longitud de paso mediante búsqueda lineal

                    x_new = x + alpha * p  # Nuevo punto

                    s = x_new - x
                    y = df(x_new) - df_val

                    rho = 1 / np.dot(y, s)

                    H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

                    x = x_new
                    iter_count += 1
                    path.append(x)

                    if np.linalg.norm(alpha * p) < tol or iter_count >= max_iter:
                        break

                return x, path

            def line_search(f, df, x, p):
                alpha = 1.0
                c = 0.9  # Factor de reducción del paso
                rho = 0.5  # Factor de reducción del tamaño de búsqueda

                while f(x + alpha * p) > f(x) + c * alpha * np.dot(df(x), p):
                    alpha *= rho

                return alpha

            # Ejemplo de uso
            def f(x):
                return x[0]**2 + x[1]**2

            def df(x):
                return np.array([2*x[0], 2*x[1]])

            x0 = np.array([1.0, 1.0])
            solution, path = bfgs(f, df, x0)

            # Crear una malla de puntos para la visualización del gráfico
            x_values = np.linspace(-2, 2, 100)
            y_values = np.linspace(-2, 2, 100)
            X, Y = np.meshgrid(x_values, y_values)
            Z = X**2 + Y**2  # Función objetivo

            # Graficar la función objetivo
            fig, ax = plt.subplots()
            ax.contour(X, Y, Z, levels=20, cmap='viridis')

            # Graficar el camino de optimización
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'ro-')

            # Graficar la solución encontrada
            ax.plot(solution[0], solution[1], 'go', label='Solución')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Optimización con Quasi-Newton (BFGS)')
            ax.legend()
            ax.grid(True)

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)

        if st.sidebar.checkbox("Ejemplo"):
            st.title("***Ejemplo***")
            st.write("Considera la siguiente funcion y encuentra el minimo de la funcion.")
            st.latex(r'f(x) = x^4-3x^2')
            st.write("Entonces:")
            st.write("Elegimos un punto inicial x0 en el dominio de la funcion.Por ejemplo x0=2.")
            st.write("Ieteramos:")
            st.write("Calculamos la direccion de busqueda utilizando la formula BFGS.")
            st.latex(r'd=-[Bk]^{-1} * \nabla f(xk)')
            st.write("Actualizamos el punto actual xk utilizando la formula:")
            st.latex(r'xk+1= xk + \alpha*d')
            st.write("Calculamos el valor de alfa utilizando algun metodo de linea como la busqueda en linea o un metodo de backtracking.")
            st.write("Actualizamos la aproximacion de la matriz Hessiana Bk+1 utilizando la formula BFGS:")
            st.latex(r'Bk+1 = Bk + \nabla B')
            st.write("Repetimos los pasos anteriores hasta que se alcance la convergencia deseada o se cumpla un criterio de parada.")
            st.write("En cada iteracion, se calcula la direccion de busqueda utilizando la aproximacion de la matriz Hessiana y el gradiente de la funcion, y se actualiza el punto actual utilizando esta direccion.")
            st.write("El proceso continua hasta que se encuentra un minimo local o se alcanza la convergencia deseada.")
            st.write("En este caso la funcion convergera rapidamente al minimo global en x=0.")
    elif metodo == "Raices de Broyden":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt

            def broyden(f, x0, tol=1e-6, max_iter=100):
                x = x0
                iter_count = 0
                path = [x]  # Lista para almacenar el camino de optimización

                while True:
                    f_val = f(x)
                    J_inv = np.linalg.pinv(jacobian(f, x))
                    dx = -np.dot(J_inv, f_val)  # Dirección de búsqueda

                    x_new = x + dx  # Nuevo punto

                    x = x_new
                    iter_count += 1
                    path.append(x)

                    if np.linalg.norm(dx) < tol or iter_count >= max_iter:
                        break

                return x, path

            def jacobian(f, x):
                n = len(x)
                J = np.zeros((n, n))
                h = 1e-6

                for i in range(n):
                    x_plus_h = x.copy()
                    x_plus_h[i] += h

                    J[:, i] = (f(x_plus_h) - f(x)) / h

                return J

            # Ejemplo de uso
            def f(x):
                return np.array([x[0]**2 - 4, x[1]**2 - 9])

            x0 = np.array([1.0, 1.0])
            solution, path = broyden(f, x0)

            # Crear una malla de puntos para la visualización del gráfico
            x_values = np.linspace(-5, 5, 100)
            y_values = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x_values, y_values)
            Z1 = X**2 - 4
            Z2 = Y**2 - 9

            # Graficar las curvas de nivel de la función
            fig, ax = plt.subplots()
            ax.contour(X, Y, Z1, levels=[0], colors='blue')
            ax.contour(X, Y, Z2, levels=[0], colors='red')

            # Graficar el camino de optimización
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'ro-')

            # Graficar la solución encontrada
            ax.plot(solution[0], solution[1], 'go', label='Solución')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Búsqueda de Raíces con Broyden')
            ax.legend()
            ax.grid(True)

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Downlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="Busqueda_Raices_Broyden.py",
                mime="text/palin")
            #Codigo funcional
            import numpy as np
            import matplotlib.pyplot as plt

            def broyden(f, x0, tol=1e-6, max_iter=100):
                x = x0
                iter_count = 0
                path = [x]  # Lista para almacenar el camino de optimización

                while True:
                    f_val = f(x)
                    J_inv = np.linalg.pinv(jacobian(f, x))
                    dx = -np.dot(J_inv, f_val)  # Dirección de búsqueda

                    x_new = x + dx  # Nuevo punto

                    x = x_new
                    iter_count += 1
                    path.append(x)

                    if np.linalg.norm(dx) < tol or iter_count >= max_iter:
                        break

                return x, path

            def jacobian(f, x):
                n = len(x)
                J = np.zeros((n, n))
                h = 1e-6

                for i in range(n):
                    x_plus_h = x.copy()
                    x_plus_h[i] += h

                    J[:, i] = (f(x_plus_h) - f(x)) / h

                return J

            # Ejemplo de uso
            def f(x):
                return np.array([x[0]**2 - 4, x[1]**2 - 9])

            x0 = np.array([1.0, 1.0])
            solution, path = broyden(f, x0)

            # Crear una malla de puntos para la visualización del gráfico
            x_values = np.linspace(-5, 5, 100)
            y_values = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x_values, y_values)
            Z1 = X**2 - 4
            Z2 = Y**2 - 9

            # Graficar las curvas de nivel de la función
            fig, ax = plt.subplots()
            ax.contour(X, Y, Z1, levels=[0], colors='blue')
            ax.contour(X, Y, Z2, levels=[0], colors='red')

            # Graficar el camino de optimización
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'ro-')

            # Graficar la solución encontrada
            ax.plot(solution[0], solution[1], 'go', label='Solución')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Búsqueda de Raíces con Broyden')
            ax.legend()
            ax.grid(True)

            # Mostrar el gráfico en Streamlit
            st.pyplot(fig)


        if st.sidebar.checkbox("Ejemplo"):
            st.title("***Ejemplo***")
            st.write("Queremos encontrar una raiz de la siguiente ecuacion no lineal:")
            st.latex(r'f(x)=x^3-2x-5=0')
            st.write("Para aplicar el metodo de Raices de Broyden, necesitamos tener una estimacion. por lo que seleccionamos a x0=2 como aproximacion inicial.")
            st.write("Evaluamos la funcion en la aproximacion inicial.")
            st.latex(r'f(x_0)= 2^3-2(2)-5=1')
            st.write("Calculamos la derivada aproximada de la funcione en x0.")
            st.latex(r'f´(x_0)=3(2)^2-2=10')
            st.write("Ahora, vamos a construir la matriz de aproximación inicial B0, que es una matriz 1x1 en este caso")
            st.latex(r'B0=[10]')
            st.write("Calculamos la siguiente aproximacion de lariz usando la formula.")
            st.latex(r'x_1 = x_0-(B_0)^{-1}*f(x_0)=2-(10)^{-1}*1=2-0.1=1.9')
            st.write("Evaluamos la funcion en la nueva aproximacion:")
            st.latex("f(x_1)=(1.9)^3-2(1.9)-5=-0.031")
            st.write("Calculamos el cambio en la funcion y en la variable:")
            st.latex(r'\delta f=f(x_1)-f(x_0)=-0.031 - 1=-1.031')
            st.latex(r'\delta x=x_1-x_0=1.9-2=-0.1')
            st.write("Actualizamos la matriz B usando la formula de actualizacion de Broyden:")
            st.latex(r'B_1=b_0+(\delta f-B_0*\delta x)*(\frac{\delta x^T}{\delta x^T * \delta x})')
            st.write("Repetimos los pasos 4-7 hasta que la aproximación converja a la raíz deseada.")
            st.write("Podemos establecer un criterio de convergencia basado en la magnitud de delta_x o delta_f.")
    elif metodo == "BFGS":
        if st.sidebar.checkbox("Codigo"):
            st.title("***Codigo***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.optimize import minimize

            def f(x):
                return x[0]**2 + 4*x[1]**2

            def grad_f(x):
                return np.array([2*x[0], 8*x[1]])

            def main():
                x_min, x_max = -5, 5
                y_min, y_max = -5, 5

                x_values = np.linspace(x_min, x_max, 100)
                y_values = np.linspace(y_min, y_max, 100)
                X, Y = np.meshgrid(x_values, y_values)
                Z = X**2 + 4*Y**2

                fig, ax = plt.subplots()
                ax.contour(X, Y, Z, levels=20, colors='gray')

                initial_points = [(-4, -3), (2, -1), (4, 3)]

                for i, initial_point in enumerate(initial_points):
                    result = minimize(f, initial_point, method='BFGS', jac=grad_f)
                    solution = result.x

                    ax.plot(initial_point[0], initial_point[1], 'ro', label='Punto Inicial' if i == 0 else None)
                    ax.plot(solution[0], solution[1], 'go', label='Solución' if i == 0 else None)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Optimización con BFGS')
                ax.legend()

                st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Downlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="BFGS.py",
                mime="text/palin")


        if st.sidebar.checkbox("Ejemplo"):
            st.title("***Ejemplo***")
            st.write("Queremos encontrar el minimo de la funcion cuadrada:")
            st.latex(r'f(x)=x^2+3x+2')
            st.write("Para aplicar el método BFGS, necesitamos una estimación inicial del mínimo. Digamos que seleccionamos x0 = -2 como nuestra aproximación inicial.")
            st.write("Evaluamos la función y su derivada en la aproximación inicial:")
            st.latex(r'f(x_0)=(-2)^2 + 3(-2)+2=0')
            st.latex(r'f´(x_0)=2(-2)+3=-1')
            st.write("Establecemos una matriz de aproximacion inicial H0, que es una matriz de identidad 1X1.")
            st.latex(r'H_0=[1]')
            st.write("Calculamos la siguiente aproximacion del minimo utilizando la formula BFGS.")
            st.latex(r'x_1=x_0-(H_0)^{-1}*f´(X_0)=-2-(1)^{-1}*-1=-1')
            st.write("Evaluamos la función y su derivada en la nueva aproximación:")
            st.latex(r'f(x_1)=(-1)^2+(3)-1+2=0')
            st.latex(r'f(x_1)=2(-1)+3=1')
            st.write("Calculamos el cambio en la funcion y en la variable:")
            st.latex(r'\delta f=f(x_1)-f(x_0)=0-0=0')
            st.latex(r'\delta x= x_1-x_0=-1-(-2)=1')
            st.write("Actualizamos la matriz H utilizando la formula BFGS:")
            st.latex(r'H_1=H_0+\frac{(\delta x*\delta x^T)}{(\delta x^T * \delta f)}')
            st.write("Repetimos los pasos 3-6 hasta que el algoritmo converja al mínimo deseado.")
            st.write("Podemos establecer un criterio de convergencia basado en la magnitud de delta_x o delta_f.")

    

    # primera sección Definiciones
    st.sidebar.subheader("Metodos")
    if metodo == "Sistema de ecuaciones":
        st.title("Sistema de ecuaciones")
        texto = "<p style='text-align:justify; text-justify: inter-word;'>Un sistema de ecuaciones es un conjunto de dos o mas ecuaciones que se deben resolver simultaneamente para encontrar soluciones comunes.</p>"
        texto += "<p style='text-align:justify; text-justify: inter-word;'>En general, un sistema de ecuaciones tiene una o mas incognitas, y la solucion es un conjunto de esas incognitas del sistema.</p>"
        texto += "<p style='text-align:justify; text-justify: inter-word;'>Los sistemas de ecuaciones pueden ser no lineales, lo que significa que uno o más de las ecuaciones no son lineales.</p>"
        texto += "<p style='text-align:justify; text-justify: inter-word;'>En este caso, la solucion puede ser mucho mas complicada, y se necesitan aproximaciones a las soluciones.</p>"
        texto += "<p style='text-align:justify; text-justify: inter-word;'>Los sistemas de ecuaciones tienen al menos 3 posibles soluciones:</p>"
        texto += "<p style='text-align:justify; text-justify: inter-word;'>1- Que la solucion exista y sea unica.</p>"
        texto += "<p style='text-align:justify; text-justify: inter-word;'>2- Que la solucion exista y hay un número infinito de soluciones.</p>"
        texto += "<p style='text-align:justify; text-justify: inter-word;'>3- Que la solución no exista.</p>"
        st.markdown(texto, unsafe_allow_html=True)

    elif metodo == "Polinomio de Taylor":
        st.title("Polinomio de Taylor")
        texto = "<p style='text-align:justify;text-justify;inter-word;'>El Polinomio de Taylor es una suma finita de derivadas locales evaluadas en un punto concreto.</p>"
        texto += "<p style='text-align:justify;text-justify;inter-word;'>Definos de la forma: </p>"
        st.markdown(texto, unsafe_allow_html=True)
        st.latex(r'f(x)\approx f(x_0)+f´(x_0)*(x-x_0)+\frac{f´´(x_0)}{2!}*(x-x_0)^2+...+\frac{f^n(x_0)}{n!}*(x-x_0)^n')
        st.latex(r'\sum_{i=0}^{n} \frac{f^n(x_0)}{n!}*(x-x_0)^n')
        st.write("Definimos:")
        st.write("f(x): función de x.")
        st.write("f(x0): función de x en un punto concreto x0.")
        st.latex(r'f^{(n)}(x_0)= f^{(n)}(x)|_{x=x_0}.')
        st.write("fn(x): n-esima derivada de la funcion f(x).")
        st.write("Aplicaciones:")
        texto="<p style='text-align:justify;text-justify:inter-word;'>Las aplicaciones genralmente son en activos y productos financieros en los que el precio se expresa como una funcion no lineal.</p>"
        texto+="<p style='text-align:justify;text-justify:inter-word;'>Por ejemplo, el precio de un titulo de deuda a corto plazo es una función no lineal que depende de los tipos de interés.</p>"
        st.markdown(texto, unsafe_allow_html=True)

    elif metodo == "Polinomio de Taylor Multivariable":
        st.title("Polinomio de Taylor Multivariable")
        texto="<p style='text-aling:justify;text-justify;inter-word'>El polinomio de Taylor Multivariable es una herramienta que permite aproximar una funcion multivariable alrededor de un punto dado mediante unaserie de terminos.</p>"
        texto+="<p style='text-aling:justify;text-justify;inter-word'>Este polinomio es util para analizar el comportamiento de la función en un entorno cerrado al punto de interes, y puede ser utilizado para hacer estimaciones numericas. </p>"
        st.markdown(texto,unsafe_allow_html=True) 
        st.write("Definimos:")
        st.latex(r'P_k(x_1,x_2,...,x_n) = \sum_{i=0}^{n}(f^k)\frac{a_1,a_2,...,a_n}{k!}(x_1-a_1)^k(x_2-a_2)(x_2-_2)^K ... (xn-an)^k')
        texto="<p style='text-aling:justify;text-justify;inter-word'><strong>Aplicaciones: </strong></p>"
        texto+="<p style='text-aling:justify;text-justify;inter-word'>El polinomio de Taylor multivariable tiene numerosas aplicaciones en áreas como la física, la ingeniería, la estadística y la economía. </p>"
        texto+="<p style='text-aling:justify;text-justify;inter-word'>En la física, por ejemplo, se utiliza para aproximar las trayectorias de partículas en campos de fuerza complejos, mientras que en la estadística se utiliza para aproximar las funciones de densidad de probabilidad de variables aleatorias. </p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodo == "Función Inversa":
        st.title("Teorema de la función inversa")
        texto="<p style='text-align:justify;text-justify;inter-word'>El teorema de la función inversa sirve para determinar la derivada inversa de una función, sin tener que calcular su inversa. </p>"
        texto+="<p style='text-align:justify;text-justify;inter-word'>El teorema dice que si una función f es derivable y su derivada en un punto x0,f´(x0) tiene inversa, entonces es aproximación de x0 y la función original también tendrá inversa. Por lo tanto: </p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.latex(r'(f^{-1})´(y_0) = \frac{1}{f´(x_0)}')
        st.write("Donde y0=f(x0).")
        st.write("Siendo f(x)=x^3 determinando f a la -1.")
        st.write("Tenemos que")
        st.latex(r'(f^{-1})´(8)=\frac{1}{f´(x_0)}')
        st.latex(r'f´(x)=3x^2')
        st.write("También se necesita saber quién es x0.")
        st.latex(r'(x_0)^3 = 8 \Rightarrow x_0 = \sqrt[3]{8} = 2')
    elif metodo == "Newton Raphson":
        st.title("Newton Raphson")
        texto = "<p style='text-aling:justify;text-justify:inter-word'>El método de Newton-Raphson es un algoritmo iterativo utilizado para encontrar aproximaciones de las raíces de una función. Es un método eficiente para encontrar raíces de funciones no lineales y se basa en la idea de aproximar la función mediante una serie de Taylor y encontrar la raíz de la aproximación lineal.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>La definición del método de Newton-Raphson es la siguiente:</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Dada una función f(x) y una estimación inicial x0 de la raíz de la función, el método de Newton-Raphson utiliza la siguiente fórmula iterativa para obtener una mejor aproximación de la raíz.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Donde:</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'> >xn es la aproximacion actual de la raiz.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'> >f(xn) es el valor de la funcion en la aproximacion actual.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'> >f´(xn) es la derivada de la funcion aproximacion actual.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>El proceso se repite iterativamente, actualizando la aproximación de la raíz en cada paso, hasta que se alcanza una precisión deseada o se obtiene una solución aceptable. </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones</strong> </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>1-Ecuaciones no lienales:</strong>Puede ser aplicado en diversas disciplinas como matemáticas, física, ingeniería, economía y ciencias de la computación, donde se necesite encontrar soluciones numéricas de ecuaciones no lineales.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2-Optimizacion:</strong> Se utiliza en la busqueda de los puntos criticos de una funcion, donde la derivada de la funcion es 0.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>3-Ajuste de curvas:</strong>Esto permite encontrar los parametros optimos de la curva que mejor se ajusta a los datos.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>4-Modelado y simulacion:</strong>En la modelizacion matematica y simulacion de sistemas complejos,el metodo es utilizado para la resolucion de ecuaciones no lineales.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>5-Analisis de estebilidad:</strong>El metodo en esta aplicacion es utilizada para encontrar los puntos de equilibrio o los valores criticos que determinan la estabilidad del sistema.</p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodo == "Newton Multivariable":
        st.title("Newton Multivariable")
        texto = "<p style='text-align:justify;text-justify:inter-word'>El método iterativo para sistemas de ecuaciones converge linealmente. Como en el método de una incógnita, pero puede crearse un método de convergencia cuadrática es decir, el método de Newton Raphson.</p>"
        texto += "<p style='text-align:justify;text-justify:inter-word'>Supóngase que se resuelve un sistema <sub>F1</sub>(x,y) y <sub>F2</sub>(x,y), donde ambas funciones son continuas y diferenciables, de modo que puedan expandirse en la serie de Taylor.</p>"
        texto += "<p style='text-align:justify;text-justify:inter-word'>Primera Aproximación:</p>"
        texto += "<p style='text-align:justify;text-justify:inter-word'>Esta se calcula primeramente sustituyendo los valores iniciales de x,y se obtiene valores h y j.</p>"
        texto += "<p style='text-align:justify;text-justify:inter-word'>Segunda iteración:</p>"
        texto += "<p style='text-align:justify;text-justify:inter-word'>Se calcula sustituyendo los nuevos valores iniciales de x,y, y obtenemos más valores de h y j.</p>"
        texto += "<p style='text-align:justify;text-justify:inter-word'>Los cuales son los nuevos valores de x y y primeramente sustituyendo los valores iniciales de x,y y obtenemos una matriz delimitada.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones:</strong></p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>1-Optimizacion:</strong> Es utilizado para encontrar puntos criticos en funciones objetivo en problemas de optimizacion.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2-Sistemas de Ecuaciones no lineales:</strong> Se busca la eficiencia con el metodo de Newton Multivariable, para buscar las soluciones de multiples ecuaciones.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>3-Ajuste de curvas:</strong>El método de Newton multivariado se utiliza en el ajuste de curvas para encontrar los parámetros óptimos que minimizan la diferencia entre los datos observados y la curva ajustada.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>4-Elemento infinitos:</strong>En la simulación numérica y el análisis de estructuras, el método de Newton multivariado se utiliza para resolver sistemas de ecuaciones no lineales que surgen en el método de elementos finitos.  </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>5-Modelado Matematico:</strong> el método de Newton multivariado se utiliza para resolver ecuaciones que describen fenómenos físicos o sistemas complejos.</p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodo == "Quasi-Newton":
        st.title("Quasi-Newton")
        texto = "<p style='text-aling:justify;text-justify:inter-word'>El metodo de Quasi-Newton es una tecnica utilizada para resolver problemas de optimizacion en los que se busca encontrar el minimo o maximo de una funciion.A difeferncia del Metodo de Newton, que requiere calculo de la matris Hessiana de la funcion objetivo,el Quasi-Newton aproxima la Hessiana.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>En lugar de calcular la matriz Hessiana exacta de cada iteracion, el metodo Quasi-Newton actualiza una matriz de aproximacion de la Hessiana utilizando la diferencia entre los gradientes de la funcion objetivo en duferentes puntos.Esta actualiizacion permite una mejor aproximacion.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones: </strong></p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>El Meto de Quasi-Newton tiene diversas aplicaciones en el campo de la resoluciones de problemas no lineales.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>1-Optimizacion de funciones:</strong> Se utiliza para encontrar los minomos o maximos de funciones no lineales, en campos como la economia, ingenireria, y ciencias sociales.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2-Ajuste de curvas:</strong> Se utiliza en el analisis de datos para dar sentido a los conjuntos de corvas observables, para despues ser explicables en modelos estadisticos y su interpolacion de datos.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>3-Programacion matematica:</strong> El metodo Quasi-Newton sw aplica en problemas de programacion matematica, donde se busca encontrar la asignacion de recursos sujetos a restricciones. </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>4-Aprendizaje automatico:</strong> En el aprendizaje automatico, se utiliza a Quasi-Newton en los algoritmos de optimizacion utilizados en el entrenamiento de modelos de aprendizaje automatico.</p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodo == "Raices de Broyden":
        st.title("Busqueda de Raices de Broyden")
        texto = "<p style='text-align:justify;text-justify:inter-word'>Este metodo es utlizadi para encontrar las raices de una funcion no lineal.Este metodo fue propuesto por CHARLES BROYDEN como una alternativa a los metodos tradicionales de busqueda de raices.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>En la busqueda de raices de BROYDEN, se busca determinar el valor de x que stisface la ecuacion f(x)=0, donde f es una funcion no lineal dde una o mas variables, lo cual es una aproximacion iterativa de la funcion y su derivada. </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Este metodo es particularmente util en casos donde calcular la derivada exacta de la funcion puede ser costoso o dificil.Al aproximar la matriz Jacobina de manera iterativa, evita la necesidad de calcular exactamente en cada iteración,llo que ahorra tiempo computacional.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones </strong></p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>1-Problemas de optimizacion:</strong> Se utiliza para encontrar los punntos criticos de funciones no lineales.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2-Analisis de sistemas no lineales:</strong> El metodo se utiliza para encontrar las soluciones de estos sistemas, lo que es util en areas como la fisica,ingenieria,estructural,economia.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>3-Modelado y simulacion:</strong> El modelo de Broyden se utiliza para obtener resultados numericos precisos.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>4-Ajuste de curvas:</strong> En analisis de datos, el metodo de Broyden puede ser utilizado para ajustar una curva a un conjunto de puntos experriimentales.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>5-Dinamica de fluidos computacional:</strong> En la simulacion numerica se pueden encontrar ecuaciones no lineales que describeb el comportamiento de flujo. </p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodo == "BFGS":
        st.title("Metodod BFGS- Broyden Fletcher Goldfarb Shanno")
        texto = "<p style='text-aling:justify;text-justify:inter-word'>Este metodo es un algoritmo de optimizacion que se utiliiza para resolver problemas de optimizacion no lineales, desarrollado en 1970. </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Este metodo tiene como finalidad encontrar el minimo de una funcion objetivo evitando la Hessiana, lo cual puede ser computacionalmente costoso, sustituyendola por una aproximacion actualizada en cada iteracion. </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>El algoritmo BFGS actualiza iterativamente la matriz aproximada de la Hessiana inversa y busca la direccion de busqueda optima para acercarse al minimo de la funcion objetivo. </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Utiliza la informacion de las iteraciones anteriores para ajustar la aproximacion de la matriz Hessiana inversa,permitiendo una convergencia mas rapida y eficiente.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones</strong></p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>1-Optimizacion:</strong> El metodo se utiliza para optimizar funciones en campos como la fisica,quimica,ingenieria estructural y mecanica de fluidos.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2-Ajustes de modelos y regresion:</strong> BFGS se utiliza para ajustar modelos matematicos a datos observados.Esto puede ncontrar los alores optimos de los coeficientes de un modelo lineal o no lineal.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>3-Aprendizaje automatico:</strong> El BFGS se utiliza para optimizar la funcion de perdida en algoritmos de entrenamiento de modelos como redes neuronales y ayuda a ajustar los pesos y parametros del modelo ara mejorar su precision y rendimiento.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>4-Rutas y logistica:</strong>En aplicaciones de planificacion y rutas logicas el BFGS se utiliza para optimizar la asignacion de rutas.Puede ayudar a minimizar costos y maximizar la eficiencia en la distribucion de bienes y servicios.</p>"
        st.markdown(texto,unsafe_allow_html=True)

    #segunda seccion Selecciones
    if metodos2 == "Lagrange":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
           import numpy as np
            import matplotlib.pyplot as plt
            import streamlit as st

            def lagrange_interpolation(x, y, x_interp):
                n = len(x)
                m = len(x_interp)
                y_interp = np.zeros(m)

                for i in range(m):
                    for j in range(n):
                        l = 1.0
                        for k in range(n):
                            if k != j:
                                l *= (x_interp[i] - x[k]) / (x[j] - x[k])
                        y_interp[i] += l * y[j]

                return y_interp

            # Ejemplo de uso
            x = np.array([0, 1, 2, 3, 4])
            y = np.array([1, 2, 4, 8, 16])

            x_interp = np.linspace(0, 4, 100)
            y_interp = lagrange_interpolation(x, y, x_interp)

            # Crear la gráfica
            fig, ax = plt.subplots()
            ax.plot(x_interp, y_interp, label='Interpolación')

            # Agregar puntos de datos originales
            ax.scatter(x, y, color='red', label='Datos originales')

            # Configurar etiquetas y título
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Interpolación Lagrange')
            ax.legend()

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt
            
            def lagrange_interpolation(x, y, x_interp):
                n = len(x)
                m = len(x_interp)
                y_interp = np.zeros(m)

                for i in range(m):
                    for j in range(n):
                        l = 1.0
                        for k in range(n):
                            if k != j:
                                l *= (x_interp[i] - x[k]) / (x[j] - x[k])
                        y_interp[i] += l * y[j]

                return y_interp

            # Ejemplo de uso
            x = np.array([0, 1, 2, 3, 4])
            y = np.array([1, 2, 4, 8, 16])

            x_interp = np.linspace(0, 4, 100)
            y_interp = lagrange_interpolation(x, y, x_interp)

            # Crear la gráfica
            fig, ax = plt.subplots()
            ax.plot(x_interp, y_interp, label='Interpolación')

            # Agregar puntos de datos originales
            ax.scatter(x, y, color='red', label='Datos originales')

            # Configurar etiquetas y título
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Interpolación Lagrange')
            ax.legend()

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            st.write("Supongamos que queremos encontrar el método de Lagrange de la función.")
            st.latex(r'f(x,y)=x^2+y^2')
            st.write("Sujeta a la restricción g(x,y)=x+y-1=0.")
            st.write("Definimos la funcion objetivo y la restricción:")
            st.latex(r'f(x,y)=x^2+y^2')
            st.latex(r'g(x,y)=x+y-1')
            st.write("Formamos la ecuación de Lagrange:")
            st.latex("L(x, y, λ) = f(x, y) - λ * g(x, y)")
            st.latex("L(x, y, λ) = x^2 + y^2 - λ(x + y - 1)")
            st.write("Calculamos las derivadas parciales de L(x,y,λ)")
            st.latex(r'∂L/∂x = 2x - λ = 0')
            st.latex(r'∂L/∂y = 2y - λ = 0')
            st.latex(r'∂L/∂λ = x + y - 1 = 0')
            st.write("Resolvemos el sistema de ecuaciones.")
            st.write("De la primera ecuación: 2x - λ = 0, tenemos x = λ/2.")
            st.write("De la segunda ecuación: 2y - λ = 0, tenemos y = λ/2.")
            st.write("Entonces, x = λ/2 = 2/2 = 1 y y = λ/2 = 2/2 = 1.")
            st.write("Sustituir los valores de x, y y λ en la función objetivo para encontrar el valor mínimo:")
            st.latex(r'f(1, 1) = 1^2 + 1^2 = 2.')
            st.write("Por lo tanto, el mínimo valor de la función")
            st.latex(r'f(x, y) = x^2 + y^2')
            st.write("Sujeto a la restricción g(x, y) = x + y - 1 = 0, es 2, y ocurre en el punto (1, 1).")

    if metodos2 == "Splines Cubicos":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
                import numpy as np
                import matplotlib.pyplot as plt
                from scipy.interpolate import CubicSpline
                import streamlit as st

                # Datos de ejemplo
                x = np.array([0, 1, 2, 3, 4])
                y = np.array([1, 2, 4, 8, 16])

                # Calcular splines cúbicos
                cs = CubicSpline(x, y)

                # Rango de valores para la interpolación
                x_interp = np.linspace(x.min(), x.max(), 100)
                y_interp = cs(x_interp)

                # Crear la gráfica
                fig, ax = plt.subplots()
                ax.plot(x_interp, y_interp, label='Interpolación')

                # Agregar puntos de datos originales
                ax.scatter(x, y, color='red', label='Datos originales')

                # Configurar etiquetas y título
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Interpolación con Splines Cúbicos')
                ax.legend()

                # Mostrar la gráfica en Streamlit
                st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            st.write("Se recomienda Descargar el codigo y ejecutatlo en el IDE de preferencia ya que streamlit no tiene soporte actual para scipy.interpolate")
            
        if st.sidebar.checkbox("Ejemplo"):
            st.title("**Ejemplo**")
            st.write("Supongamos que tenemos las siguientes puntos de datos:")
            st.write("Punto 1: (0,1)")
            st.write("Punto 2: (1,4)")
            st.write("Punto 3: (2,2)")
            st.write("Punto 4: (3,6)")
            st.write("Queremos encontrar una curva suave que pase por estos puntos utilizados splines cúbicos:")
            st.write("1-Dividir el dominio en segmentos dividiremos el dominio en segmentos entre los puntos de datos. En este caso, tendremos tres segmentos: [0,1], [1,2] y [2,3].")
            st.write("2-  Definir polinomios cúbicos dentro de cada segmento, definiremos polinomios cúbicos para aproximar la curva. En este caso, cada segmento se puede representar con un polinomio de la forma:")
            st.latex(r'S(x) = a + b * (x - xi) + c * (x - xi)^2 + d * (x - xi)^3')
            st.write("donde a, b, c y d son coeficientes desconocidos y xi es el punto inicial del segmento.")
            st.write("3-  Determinar los coeficientes para determinar los coeficientes de los polinomios cúbicos, necesitamos establecer condiciones en los puntos de interpolación. ")
            st.write("Utilizaremos las siguientes condiciones:")
            st.write("> Interpolación: La curva debe pasar por los puntos de datos.")
            st.write("> Continuidad de la primera derivada: La primera derivada de los polinomios adyacentes debe ser igual en los puntos de intersección.")
            st.write("> Continuidad de la segunda derivada: La segunda derivada de los polinomios adyacentes debe ser igual en los puntos de intersección.")
            st.write("4- Resolver el sistema de ecuaciones con las condiciones establecidas, podemos formar un sistema de ecuaciones y resolverlo para encontrar los coeficientes desconocidos. ")
            st.latex(r'1+b1+c1+d1 = 4')
            st.latex(r'b1+2c1+3d1 = b2')
            st.latex(r'2c1 + 6d1 = 2c2')
            st.write("Repetimos este proceso para los otros segmentos.")
            st.write("Una vez que se encuentren los coeficientes para todos los segmentos, podemos utilizar los polinomios cúbicos resultantes para interpolar la curva entre los puntos de datos.")
            st.write("En el primer segmento [0, 1]:")
            st.latex(r'S1(x) = 1 + 3x + (-3)x^2 + 1x^3')
            st.write("En el segundo segmento [1, 2]:")
            st.latex(r'S2(x) = 4 + (-2)x + (-3)x^2 + 1x^3')
            st.write("En el tercer segmento [2, 3]:")
            st.latex(r'S3(x) = 2 + 4x + (-6)x^2 + 3x^3')
            st.write("Puedes evaluar estos polinomios en cualquier valor de x dentro del rango correspondiente para obntener la estimación de la funcion interpolada.")

    if metodos2 == "Hermite":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
                import numpy as np
                import matplotlib.pyplot as plt
                import streamlit as st

                # Datos de ejemplo
                x = np.array([0, 1, 2, 3, 4])
                y = np.array([1, 2, 4, 8, 16])
                dy_dx = np.array([1, 3, 6, 12, 24])

                # Función para calcular la interpolación de Hermite
                def hermite_interpolation(x, y, dy_dx, x_interp):
                    n = len(x)
                    m = len(x_interp)
                    y_interp = np.zeros(m)

                    for i in range(m):
                        for j in range(n-1):
                            if x[j] <= x_interp[i] <= x[j+1]:
                                t = (x_interp[i] - x[j]) / (x[j+1] - x[j])
                                h00 = (1 + 2*t) * (1 - t)**2
                                h10 = t * (1 - t)**2
                                h01 = t**2 * (3 - 2*t)
                                h11 = t**2 * (t - 1)
                                y_interp[i] = h00 * y[j] + h10 * (x[j+1] - x[j]) * dy_dx[j] + h01 * y[j+1] + h11 * (x[j+1] - x[j]) * dy_dx[j+1]
                                break

                    return y_interp

                # Rango de valores para la interpolación
                x_interp = np.linspace(x.min(), x.max(), 100)
                y_interp = hermite_interpolation(x, y, dy_dx, x_interp)

                # Crear la gráfica
                fig, ax = plt.subplots()
                ax.plot(x_interp, y_interp, label='Interpolación')

                # Agregar puntos de datos originales
                ax.scatter(x, y, color='red', label='Datos originales')

                # Configurar etiquetas y título
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title('Interpolación de Hermite')
                ax.legend()

                # Mostrar la gráfica en Streamlit
                st.pyplot(fig)

            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt
            

            # Datos de ejemplo
            x = np.array([0, 1, 2, 3, 4])
            y = np.array([1, 2, 4, 8, 16])
            dy_dx = np.array([1, 3, 6, 12, 24])

            # Función para calcular la interpolación de Hermite
            def hermite_interpolation(x, y, dy_dx, x_interp):
                n = len(x)
                m = len(x_interp)
                y_interp = np.zeros(m)

                for i in range(m):
                    for j in range(n-1):
                        if x[j] <= x_interp[i] <= x[j+1]:
                            t = (x_interp[i] - x[j]) / (x[j+1] - x[j])
                            h00 = (1 + 2*t) * (1 - t)**2
                            h10 = t * (1 - t)**2
                            h01 = t**2 * (3 - 2*t)
                            h11 = t**2 * (t - 1)
                            y_interp[i] = h00 * y[j] + h10 * (x[j+1] - x[j]) * dy_dx[j] + h01 * y[j+1] + h11 * (x[j+1] - x[j]) * dy_dx[j+1]
                            break

                return y_interp

            # Rango de valores para la interpolación
            x_interp = np.linspace(x.min(), x.max(), 100)
            y_interp = hermite_interpolation(x, y, dy_dx, x_interp)

            # Crear la gráfica
            fig, ax = plt.subplots()
            ax.plot(x_interp, y_interp, label='Interpolación')

            # Agregar puntos de datos originales
            ax.scatter(x, y, color='red', label='Datos originales')

            # Configurar etiquetas y título
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Interpolación de Hermite')
            ax.legend()

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            st.write("Tenemos el polinomio x^2+2x+2 no tiene raices reales. Aplicamos la formula de Hermite.")
            st.latex(r'\int \frac{P(x)}{Q(x)}dx = \frac{P_1(x)}{Q_1(x)} + \int \frac{P_2(x)}{Q_2(x)}dx')
            st.write("Entonces Q(x)=(x^2+2x+2). Entonces:")
            st.latex(r'Q(x)=mcd{(x^2+2x+2)^2,2(x^2+1)(2x+2)}=x^2+2x+2')
            st.latex(r'Q_2(x)=\frac{(x^2+2x+2)^2}{x^2+2x+2} = x^2+2x+2')
            st.write("La igualdad, se transforma en:")
            st.latex(r'\int {3+5}{(x^2+2x+2)^2}dx = \frac{Ac+B}{x^2+2x+2}+\int {Cx+D}{x^2+2x+2}dx.')
            st.write("Derivando, obtenemos:")
            st.latex(r'\frac{3x+5}{(x^2+2x+2)} = \frac{A(x^2+2x+2)-(2x+2)(Ax+B)}{(x^2+2x+2)}+\frac{Cx+D}{(x^2+2x+2)}')
            st.write("Es igual:")
            st.latex(r'\frac{A(x^2+2x+2)-(2x-2)(Ax+B)(Cx+D)(x^2+2x+2)}{(x^2+2x+2)}')
            st.write("Igualando los numeradores, identificando coeficientes y resolviendo el sistema tenemos:")
            st.latex(r'\frac {2x-1}{4(x^2+2x+2)} + arctan(x+1)+C')
    if metodos2 == "OLS":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt
            import streamlit as st

            # Datos de ejemplo
            x = np.array([0, 1, 2, 3, 4])
            y = np.array([1, 3, 5, 4, 2])

            # Calcular los coeficientes de la recta de regresión
            n = len(x)
            X = np.column_stack((np.ones(n), x))
            coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

            # Calcular los valores ajustados
            y_pred = X @ coefficients

            # Crear la gráfica
            fig, ax = plt.subplots()
            ax.scatter(x, y, color='red', label='Datos')
            ax.plot(x, y_pred, label='Ajuste OLS')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Mínimos Cuadrados Ordinarios (OLS)')
            ax.legend()

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt

            # Datos de ejemplo
            x = np.array([0, 1, 2, 3, 4])
            y = np.array([1, 3, 5, 4, 2])

            # Calcular los coeficientes de la recta de regresión
            n = len(x)
            X = np.column_stack((np.ones(n), x))
            coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

            # Calcular los valores ajustados
            y_pred = X @ coefficients

            # Crear la gráfica
            fig, ax = plt.subplots()
            ax.scatter(x, y, color='red', label='Datos')
            ax.plot(x, y_pred, label='Ajuste OLS')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Mínimos Cuadrados Ordinarios (OLS)')
            ax.legend()

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            st.write("El metodo OLS es un enfoque de regresion lineal que busca la minimizar la suma de los errores al cuadrado entre los valores observados.")
            st.write("Supongamos que tenemos un conjunto de datos con una variable independiente x y una variable dependiente y. Queremos ajustar una línea recta a estos datos utilizando el método OLS.")
            st.write("Tomemos los siguientes datos.")
            st.latex(r'(x1, y1) = (1, 3)')
            st.latex(r'(x2, y2) = (2, 5)')
            st.latex(r'(x3, y3) = (3, 7)')
            st.latex(r'(x4, y4) = (4, 9)')
            st.write("El modelo de regresión lineal tiene la forma:")
            st.latex(r'y = \beta 0 + \beta 1x')
            st.write("Donde β0 es la intersección y β1 es la pendiente de la línea recta.")
            st.write("El objetivo del método OLS es encontrar los valores de β0 y β1 que minimicen la suma de los errores al cuadrado entre los valores observados y los valores predichos por el modelo.")
            st.write("La fórmula para calcular los coeficientes β0 y β1 mediante OLS es:")
            st.latex(r'\beta 1 =  \frac{(\sum xi-x)(yi-y)}{\sum ((xi-x)^2)}')
            st.latex(r'\beta = y - \beta 1x')
            st.write("Calculando los valores, obtenemos:")
            st.latex(r'x̄ = \frac{(1 + 2 + 3 + 4)}{4} = 2.5')
            st.latex(r'ȳ = \frac{(3 + 5 + 7 + 9)}{ 4} = 6')
            st.latex(r'Σ((xi - x̄)(yi - ȳ)) = (1 - 2.5)(3 - 6) + (2 - 2.5)(5 - 6) + (3 - 2.5)(7 - 6) + (4 - 2.5)(9 - 6) = -3.5')
            st.latex(r'Σ((xi - x̄)²) = (1 - 2.5)² + (2 - 2.5)² + (3 - 2.5)² + (4 - 2.5)² = 5')
            st.write("Sustituyendo estos valores en las fórmulas, obtenemos:")
            st.latex(r'β1 = \frac{-3.5}{5} = -0.7')
            st.latex(r'β0 = 6 - (-0.7) * 2.5 = 7.75')
            st.write("Entonces, la línea de regresión ajustada mediante OLS es:")
            st.latex(r'y = 7.75 - 0.7x')

    #Segunda seccion Definiciones
    if metodos2 == "Lagrange":
        st.title("Lagrange")
        texto = "<p style='text-align:justify; text-justify: inter-word;'>El método de Lagrange es una reformulación del polinomio de Newton que evita los cálculos de las diferencias divididas.</p>"
        st.markdown(texto, unsafe_allow_html=True)

        texto = "<p style='text-align:justify;text-justify:inter-word;'>Los polinomios de Lagrange ocurren cuando tenemos distintos (x0,y0),(x1,y1) y buscamos un polinomio que se ajuste a un punto desconocido.</p>"
        st.markdown(texto, unsafe_allow_html=True)

        st.latex(r'P(x) = \frac{x-x_1}{x_0-x_1}y0 + \frac{x - x_0}{x_1-x_0} = L_0(x)y_0 + L_1(x)y_1')

        st.write("Cuando x = x0:")
        st.latex(r'L_0(x_0)=1 , L_1(x_0)=0')
        st.latex(r'L_0(x_1)=0 , l1(x_1)=1')

        texto = "<p style='text-align:justify;text-justify:inter-word;'>Entendemos entonces que entre más puntos tengamos es necesario que en la posición xk donde k=0,1,...,n debemos omitir esa parte para que no se vuelva cero la ecuación. </p>"
        st.markdown(texto, unsafe_allow_html=True)

        st.latex(r'L_k(x)=\frac{(x-x_0)(x-x_1)...(x-x_{k-1})(x-x_{k+1})...(x-x_n)}{(x_k-x_0)(x_k-x_1)...(x_k-x_{k-1})(x_k-x_{k+1})...(x_k-x_n)}')

        st.latex(r'\prod_{i=1}^{n} x_i = \frac{x-x_i}{x_k-x_i}')

        st.write("**Aplicaciones**")
        texto = "<p style='text-align:justify;text-justify:inter-word;'><strong>Física:</strong> Se utiliza en la interpolación de datos experimentales para encontrar una función que represente el comportamiento de un sistema físico</p>"
        st.markdown(texto, unsafe_allow_html=True)

        texto = "<p style='text-align:justify;text-justify:inter-word;'><strong>Ciencias Ambientales:</strong> En las ciencias ambientales, el polinomio de interpolación de Lagrange se utiliza para modelar y predecir la calidad del aire y del agua, la evolución de los ecosistemas y la propagación de contaminantes</p>"
        st.markdown(texto, unsafe_allow_html=True)

    if metodos2 == "Splines Cubicos":
        st.title("Splines Cubicos")
        texto="<p style='text-aling:justify;text-aling:inter-word;'>Los splines (Trazadores) cúbicos son una manera de ajustar un conjunto de datos a través de una curva suave.</p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'>Un spline cúbico es una curva suave definida por una serie de segmentos cúbicos consecutivos. Estos segmentos se unen en puntos llamados *nodos* y se ajustan de manera que la curva resultante sea continua y diferenciable.</p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'> La fórmula del spline cúbico se puede expresar de la siguiente manera:</p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.latex(r'S(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3')
        st.write("Donde:")
        st.write("S(x) representa la función del spline cúbico.")
        st.write("a_i, b_i, c_i, d_i son coeficientes específicos para cada segmento.")
        st.write("x_i es la posición del nodo inicial del segmento.")
        st.write("**Aplicacciones:**")
        texto="<p style='text-aling:justify;text-aling:inter-word;'><strong>1-Graficos y animaciones:</strong>Los splines cúbicos se utilizan comúnmente para representar curvas suaves en gráficos y animaciones computarizadas. </p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'><strong>2-Diseño de productos:</strong> En el diseño industrial y de productos, los splines cúbicos se utilizan para modelar superficies y curvas suaves en el diseño de automóviles, muebles, productos electrónicos y otros objetos. </p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'><strong>3-Interpolación de datos:</strong>Los splines cúbicos se utilizan para interpolar y ajustar datos experimentales o puntos de control.</p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'><strong>4-Reconstrucción de imagenes médicas:</strong>En la reconstrucción de imágenes médicas, los splines cúbicos se utilizan para suavizar y reconstruir curvas y superficies a partir de datos de imagen.</p>"
        st.markdown(texto,unsafe_allow_html=True)

    if metodos2 == "Hermite":
        st.title("Hermite")
        texto="<p style='text-aling:justify;text-aling:inter-word;'>Los polinomios osculantes, representan una genralización de los polinomios de Taylor y negativos con n+1 números distintos en x0,x1,..,xn en [a,b] y los enteros no negativos.</p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'>El polinomio osculante que aproxima una funci ́on f ∈ Cm[a, b] en xi para cada i=0,1,..., n, es el polinomio de menor grado que concuerda con la fuci ́on f y con todas sus derivadas de orden menor o igual que mi en xi para cada i=0,1,...,n.</p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.write("Formula:")
        st.latex("H_{2n+1} = \sum_{j=0}^{n} f(x_j)H_n,j(x) + \sum_{j=0}^{n}f´(x_j)H´n,j(x)")
        st.write("Donde:")
        st.write("W_i(x) es una función que se utiliza para ajustar la influencia de cada punto en la interpolación.")
        st.write("L_i(x) es una función que se utiliza para interpolar los valores en función de las diferencias divididas de los puntos y las derivadas en esos puntos.")
        st.write("**Aplicaciones:**")
        texto="<p style='text-aling:justify;text-aling:inter-word;'><strong>1-Gráficos por computadora:</strong>El método de interpolación de Hermite se utiliza en la generación de gráficos por computadora para crear curvas suaves y realistas en la representación de objetos en 2D y 3D. </p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'><strong>2-Animación por computadora:</strong> En la animación por computadora, el método de interpolación de Hermite se utiliza para generar trayectorias suaves y naturales para el movimiento de personajes y objetos en la pantalla.</p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'><strong>3-Procesamiento de imagenes y visión por computadora:</strong>En el procesamiento de imágenes y la visión por computadora, el método de interpolación de Hermite se utiliza para suavizar bordes y curvas, así como para la interpolación de valores en imágenes.</p>"
        st.markdown(texto,unsafe_allow_html=True)

    if metodos2 == "OLS":
        st.title("Minimos Cuadrados Ordinarios")
        texto="<p style='text-aling:justify;text-aling:inter-word;'>El método de Mínimos Cuadrados Ordinarios (OLS) es un enfoque estadístico para estimar los parámetros desconocidos de un modelo de regresión lineal. </p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'>Se basa en minimizar la suma de los errores cuadráticos entre los valores observados y los valores predichos por el modelo lineal.</p>"
        texto="<p style='text-aling:justify;text-aling:inter-word;'>En un modelo de regresión lineal simple, la fórmula para estimar los coeficientes utilizando el método OLS es la siguiente:</p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.latex(r'\beta = (X^TX)^{-1}X^TY')
        st.write("Donde:")
        st.write("β̂ representa los coeficientes estimados del modelo.")
        st.write("X es la matriz de variables predictoras.")
        st.write("Y es el vector de valores observados de la variable dependiente.")
        st.write("**Aplicaciones+**")
        texto="<p style='text-aling:justify;text-aling:inter-word;'><strong>1-Analisis de regresión lineal:</strong> El método OLS es utilizado para estimar los coeficientes de un modelo de regresión lineal y analizar la relación entre las variables involucradas.</p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'><strong>2-Pronósticos y Predicciones:</strong>Es posible hacer pronósticos y predicciones sobre los valores futuros de la variable dependiente.</p>"
        texto+="<p style='text-aling:justify;text-aling:inter-word;'><strong>3-Analisis económico:</strong>El análisis económico para estudiar la relación entre variables económicas, como el crecimiento del PIB y el desempleo.</p>"
        st.markdown(texto,unsafe_allow_html=True)


    #tercera seccion Selecciones
    if metodos3 == "Minimos Cuadrados":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt
            

            # Datos de ejemplo
            x = np.array([0, 1, 2, 3, 4])
            y = np.array([1, 3, 5, 4, 2])

            # Calcular los coeficientes de la recta de regresión
            n = len(x)
            A = np.column_stack((x, np.ones(n)))
            coefficients = np.linalg.inv(A.T @ A) @ A.T @ y

            # Calcular los valores ajustados
            y_pred = A @ coefficients

            # Crear la gráfica
            fig, ax = plt.subplots()
            ax.scatter(x, y, color='red', label='Datos')
            ax.plot(x, y_pred, label='Ajuste Mínimos Cuadrados')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Mínimos Cuadrados')
            ax.legend()

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt
            

            # Datos de ejemplo
            x = np.array([0, 1, 2, 3, 4])
            y = np.array([1, 3, 5, 4, 2])

            # Calcular los coeficientes de la recta de regresión
            n = len(x)
            A = np.column_stack((x, np.ones(n)))
            coefficients = np.linalg.inv(A.T @ A) @ A.T @ y

            # Calcular los valores ajustados
            y_pred = A @ coefficients

            # Crear la gráfica
            fig, ax = plt.subplots()
            ax.scatter(x, y, color='red', label='Datos')
            ax.plot(x, y_pred, label='Ajuste Mínimos Cuadrados')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Mínimos Cuadrados')
            ax.legend()

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            st.write("Supongamos que tienes un conjunto de datos que representa la cantidad de horas de estudio (X) y la calificación obtenida en un examen (Y) para varios estudiantes.")
            st.write("Quieres determinar una relación lineal entre las horas de estudio y las calificaciones y encontrar la mejor línea de regresión que se ajuste a los datos.")
            st.write("Los datos muestra son los siguientes: x=2,3,4,5,6 y y=78,87,92,89,95")
            st.write("Calcular las sumas de X, Y, XY y X^2:")
            st.latex(r'Suma de X: 2 + 3 + 4 + 5 + 6 = 20')
            st.write("Suma de Y: 78 + 87 + 92 + 89 + 95 = 441")
            st.write(r'Suma de XY (producto de X e Y): (278) + (387) + (492) + (589) + (6*95) = 2067')
            st.latex(r'Suma de X^2 (cuadrado de X): (2^2) + (3^2) + (4^2) + (5^2) + (6^2) = 70')
            st.write("Calcular los coeficientes de la línea de regresión:")
            st.latex(r'Pendiente (a): a = (n * ΣXY - ΣX * ΣY) / (n * ΣX^2 - (ΣX)^2)')
            st.latex(r'Intercepto (b): b = (ΣY - a * ΣX) / n')
            st.write("En este caso, n es el número de datos (5).")
            st.write("Aplicando la fórmula:")
            st.write("a = (5 * 2067 - 20 * 441) / (5 * 70 - 20^2) ≈ 2.4")
            st.write("b = (441 - 2.4 * 20) / 5 ≈ 71.6")
            st.write("Y ≈ 2.4X + 71.6")
    if metodos3 == "Descenso Gradiente":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt
            

            def gradient_descent(X, y, learning_rate=0.01, num_iterations=100):
                n = len(X)
                theta = np.zeros(X.shape[1])  # Inicializar los parámetros en ceros
                history = []  # Lista para almacenar el historial de los valores de theta en cada iteración

                for _ in range(num_iterations):
                    gradient = (1/n) * X.T @ (X @ theta - y)  # Calcular el gradiente
                    theta = theta - learning_rate * gradient  # Actualizar los parámetros
                    history.append(theta)  # Agregar los valores de theta al historial

                return theta, history

            # Datos de ejemplo
            X = np.array([1, 2, 3, 4, 5])
            y = np.array([5, 7, 9, 11, 13])

            # Agregar columna de unos para el término independiente
            X = np.column_stack((np.ones(X.shape[0]), X))

            # Ejecutar el descenso de gradiente
            theta, history = gradient_descent(X, y)

            # Crear la gráfica de la función objetivo en cada iteración
            fig, ax = plt.subplots()
            ax.set_xlabel('Iteración')
            ax.set_ylabel('Valor de la función objetivo')
            ax.set_title('Descenso de Gradiente')

            function_values = [np.sum((X @ theta - y) ** 2) / (2 * len(X)) for theta in history]
            ax.plot(range(len(history)), function_values)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt
            

            def gradient_descent(X, y, learning_rate=0.01, num_iterations=100):
                n = len(X)
                theta = np.zeros(X.shape[1])  # Inicializar los parámetros en ceros
                history = []  # Lista para almacenar el historial de los valores de theta en cada iteración

                for _ in range(num_iterations):
                    gradient = (1/n) * X.T @ (X @ theta - y)  # Calcular el gradiente
                    theta = theta - learning_rate * gradient  # Actualizar los parámetros
                    history.append(theta)  # Agregar los valores de theta al historial

                return theta, history

            # Datos de ejemplo
            X = np.array([1, 2, 3, 4, 5])
            y = np.array([5, 7, 9, 11, 13])

            # Agregar columna de unos para el término independiente
            X = np.column_stack((np.ones(X.shape[0]), X))

            # Ejecutar el descenso de gradiente
            theta, history = gradient_descent(X, y)

            # Crear la gráfica de la función objetivo en cada iteración
            fig, ax = plt.subplots()
            ax.set_xlabel('Iteración')
            ax.set_ylabel('Valor de la función objetivo')
            ax.set_title('Descenso de Gradiente')

            function_values = [np.sum((X @ theta - y) ** 2) / (2 * len(X)) for theta in history]
            ax.plot(range(len(history)), function_values)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            st.write("Supongamos que tienes un conjunto de datos que representa la cantidad de horas de estudio (X) y la calificación obtenida en un examen (Y) para varios estudiantes.")
            st.write("Quieres determinar una relación lineal entre las horas de estudio y las calificaciones y encontrar la mejor línea de regresión que se ajuste a los datos.")
            st.latex(r'x=2,3,4,5,6')
            st.latex(r'y=78,87,92,89,95')
            st.write("Calcular las sumas de X, Y, XY y X^2:")
            st.write("Suma de X: 2 + 3 + 4 + 5 + 6 = 20")
            st.write("Suma de Y: 78 + 87 + 92 + 89 + 95 = 441")
            st.write("Suma de XY (producto de X e Y): (278) + (387) + (492) + (589) + (6*95) = 2067")
            st.write("Suma de X^2 (cuadrado de X): (2^2) + (3^2) + (4^2) + (5^2) + (6^2) = 70")
            st.write("Calcular los coeficientes de la línea de regresión:")
            st.latex(r'Pendiente (a): a = (n * ΣXY - ΣX * ΣY) / (n * ΣX^2 - (ΣX)^2)')
            st.latex(r'Intercepto (b): b = (ΣY - a * ΣX) / n')
            st.write("En este caso, n es el número de datos (5).")
            st.latex(r'a = (5 * 2067 - 20 * 441) / (5 * 70 - 20^2) ≈ 2.4')
            st.latex(r'b = (441 - 2.4 * 20) / 5 ≈ 71.6')
            st.write("Por lo tanto, la línea de regresión resultante es:")
            st.latex(r'Y ≈ 2.4X + 71.6')
    if metodos3 == "Batch y MiniBatch":
        if st.sidebar.checkbox("Código"):
            st.title("***Batch***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt

            def batch_gradient_descent(X, y, learning_rate=0.01, num_iterations=100):
                n = len(X)
                theta = np.zeros(X.shape[1])  # Inicializar los parámetros en ceros
                history = []  # Lista para almacenar el historial de los valores de theta en cada iteración

                for _ in range(num_iterations):
                    gradient = (1/n) * X.T @ (X @ theta - y)  # Calcular el gradiente
                    theta = theta - learning_rate * gradient  # Actualizar los parámetros
                    history.append(theta)  # Agregar los valores de theta al historial

                return theta, history

            # Datos de ejemplo
            X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
            y = np.array([5, 7, 9, 11, 13])

            # Ejecutar el descenso de gradiente por lotes
            theta, history = batch_gradient_descent(X, y)

            # Crear la gráfica de la función objetivo en cada iteración
            fig, ax = plt.subplots()
            ax.set_xlabel('Iteración')
            ax.set_ylabel('Valor de la función objetivo')
            ax.set_title('Descenso de Gradiente por Lotes')

            function_values = [np.sum((X @ theta - y) ** 2) / (2 * len(X)) for theta in history]
            ax.plot(range(len(history)), function_values)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            st.title("Batch")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt

            def batch_gradient_descent(X, y, learning_rate=0.01, num_iterations=100):
                n = len(X)
                theta = np.zeros(X.shape[1])  # Inicializar los parámetros en ceros
                history = []  # Lista para almacenar el historial de los valores de theta en cada iteración

                for _ in range(num_iterations):
                    gradient = (1/n) * X.T @ (X @ theta - y)  # Calcular el gradiente
                    theta = theta - learning_rate * gradient  # Actualizar los parámetros
                    history.append(theta)  # Agregar los valores de theta al historial

                return theta, history

            # Datos de ejemplo
            X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
            y = np.array([5, 7, 9, 11, 13])

            # Ejecutar el descenso de gradiente por lotes
            theta, history = batch_gradient_descent(X, y)

            # Crear la gráfica de la función objetivo en cada iteración
            fig, ax = plt.subplots()
            ax.set_xlabel('Iteración')
            ax.set_ylabel('Valor de la función objetivo')
            ax.set_title('Descenso de Gradiente por Lotes')

            function_values = [np.sum((X @ theta - y) ** 2) / (2 * len(X)) for theta in history]
            ax.plot(range(len(history)), function_values)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            st.title("***MiniBatch***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt
            

            def minibatch_gradient_descent(X, y, learning_rate=0.01, batch_size=32, num_iterations=100):
                n = len(X)
                num_batches = int(np.ceil(n / batch_size))
                theta = np.zeros(X.shape[1])  # Inicializar los parámetros en ceros
                history = []  # Lista para almacenar el historial de los valores de theta en cada iteración

                for _ in range(num_iterations):
                    indices = np.random.permutation(n)  # Generar índices aleatorios para formar los mini lotes
                    X_shuffled = X[indices]
                    y_shuffled = y[indices]

                    for i in range(num_batches):
                        start = i * batch_size
                        end = start + batch_size
                        X_batch = X_shuffled[start:end]
                        y_batch = y_shuffled[start:end]

                        gradient = (1 / batch_size) * X_batch.T @ (X_batch @ theta - y_batch)  # Calcular el gradiente
                        theta = theta - learning_rate * gradient  # Actualizar los parámetros

                    history.append(theta)  # Agregar los valores de theta al historial

                return theta, history

            # Datos de ejemplo
            X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
            y = np.array([5, 7, 9, 11, 13])

            # Ejecutar el descenso de gradiente por mini lotes
            theta, history = minibatch_gradient_descent(X, y)

            # Crear la gráfica de la función objetivo en cada iteración
            fig, ax = plt.subplots()
            ax.set_xlabel('Iteración')
            ax.set_ylabel('Valor de la función objetivo')
            ax.set_title('Descenso de Gradiente por Mini Lotes')

            function_values = [np.sum((X @ theta - y) ** 2) / (2 * len(X)) for theta in history]
            ax.plot(range(len(history)), function_values)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            st.title("MiniBatch")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt
            

            def minibatch_gradient_descent(X, y, learning_rate=0.01, batch_size=32, num_iterations=100):
                n = len(X)
                num_batches = int(np.ceil(n / batch_size))
                theta = np.zeros(X.shape[1])  # Inicializar los parámetros en ceros
                history = []  # Lista para almacenar el historial de los valores de theta en cada iteración

                for _ in range(num_iterations):
                    indices = np.random.permutation(n)  # Generar índices aleatorios para formar los mini lotes
                    X_shuffled = X[indices]
                    y_shuffled = y[indices]

                    for i in range(num_batches):
                        start = i * batch_size
                        end = start + batch_size
                        X_batch = X_shuffled[start:end]
                        y_batch = y_shuffled[start:end]

                        gradient = (1 / batch_size) * X_batch.T @ (X_batch @ theta - y_batch)  # Calcular el gradiente
                        theta = theta - learning_rate * gradient  # Actualizar los parámetros

                    history.append(theta)  # Agregar los valores de theta al historial

                return theta, history

            # Datos de ejemplo
            X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
            y = np.array([5, 7, 9, 11, 13])

            # Ejecutar el descenso de gradiente por mini lotes
            theta, history = minibatch_gradient_descent(X, y)

            # Crear la gráfica de la función objetivo en cada iteración
            fig, ax = plt.subplots()
            ax.set_xlabel('Iteración')
            ax.set_ylabel('Valor de la función objetivo')
            ax.set_title('Descenso de Gradiente por Mini Lotes')

            function_values = [np.sum((X @ theta - y) ** 2) / (2 * len(X)) for theta in history]
            ax.plot(range(len(history)), function_values)

            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)
            
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            texto = "<p style='text-aling:justify;text-justify:inter-word'><strong>Batch</strong></p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>En el enfoque de batch, se utiliza el conjunto completo de datos de entrenamiento en cada iteración del algoritmo de entrenamiento.</p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>Por ejemplo, si establecemos el tamaño de batch en 1000, dividimos nuestro conjunto de datos en 10 lotes de 1000 imágenes cada uno. </p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>Durante una iteración del entrenamiento, se alimenta el lote completo de 1000 imágenes al modelo y se calcula el error promedio en función de todas las imágenes del lote.</p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>Luego, se actualizan los pesos del modelo utilizando el algoritmo de optimización correspondiente (como el descenso de gradiente).</p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>Este proceso se repite hasta que se hayan procesado todos los lotes.</p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>MiniBatch</strong></p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>En el enfoque de minibatch, se utiliza un subconjunto más pequeño de datos de entrenamiento en cada iteración.</p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>Por ejemplo, si establecemos el tamaño de minibatch en 100, dividimos nuestro conjunto de datos en 100 lotes de 100 imágenes cada uno.</p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>Durante una iteración del entrenamiento, se alimenta un lote de 100 imágenes al modelo y se calcula el error promedio en función de esas imágenes. Luego, se actualizan los pesos del modelo.</p>"
            texto += "<p style='text-aling:justify;text-justify:inter-word'>Este proceso se repite hasta que se hayan procesado todos los lotes.</p>"
            st.markdown(texto,unsafe_allow_html=True)
    if metodos3 == "Regla Trapezoidal":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt

            def trapezoidal_rule(f, a, b, n):
                h = (b - a) / n
                x = np.linspace(a, b, n + 1)
                y = f(x)
                area = (h / 2) * (np.sum(y) - y[0] - y[-1])
                return area

            # Función de ejemplo
            def f(x):
                return np.sin(x)

            # Intervalo de integración
            a = 0
            b = np.pi

            # Número de subintervalos
            n = 100

            # Calcular el área aproximada utilizando la regla del trapecio
            area = trapezoidal_rule(f, a, b, n)

            # Crear la gráfica de la función y el área aproximada
            x = np.linspace(a, b, 1000)
            y = f(x)

            fig, ax = plt.subplots()
            ax.plot(x, y, label='Función')
            ax.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.3, label='Área Aproximada')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Regla del Trapecio')
            ax.legend()

            # Mostrar el área aproximada en Streamlit
            st.write("Área Aproximada:", area)
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt

            def trapezoidal_rule(f, a, b, n):
                h = (b - a) / n
                x = np.linspace(a, b, n + 1)
                y = f(x)
                area = (h / 2) * (np.sum(y) - y[0] - y[-1])
                return area

            # Función de ejemplo
            def f(x):
                return np.sin(x)

            # Intervalo de integración
            a = 0
            b = np.pi

            # Número de subintervalos
            n = 100

            # Calcular el área aproximada utilizando la regla del trapecio
            area = trapezoidal_rule(f, a, b, n)

            # Crear la gráfica de la función y el área aproximada
            x = np.linspace(a, b, 1000)
            y = f(x)

            fig, ax = plt.subplots()
            ax.plot(x, y, label='Función')
            ax.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.3, label='Área Aproximada')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Regla del Trapecio')
            ax.legend()

            # Mostrar el área aproximada en Streamlit
            st.write("Área Aproximada:", area)
            st.pyplot(fig)
            
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            st.write("Supongamos que queremos aproximar el valor de la integral definida de la función f(x) = x^2 en el intervalo [0, 2] utilizando la regla trapezoidal.")
            st.write("Dividir el intervalo [0, 2] en subintervalos. En este ejemplo, utilizaremos un solo subintervalo.")
            st.write("Calcular el ancho del subintervalo.")
            st.write("El ancho del subintervalo (h) se calcula dividiendo la longitud total del intervalo entre el número de subintervalos.")
            st.write("En nuestro caso, h = (2 - 0) / 1 = 2.")
            st.write("Evaluar la función en los puntos extremos del subintervalo.")
            st.write("Evaluamos la función f(x) en los puntos extremos del subintervalo. ")
            st.write("En nuestro caso, evaluamos f(0) = 0 y f(2) = 4.")
            st.write(": Aplicar la fórmula de la regla trapezoidal.")
            st.write("La fórmula de la regla trapezoidal es la siguiente:")
            st.latex(r'Integral aproximada = (h / 2) * (f(a) + f(b))')
            st.write("Integral aproximada = (2 / 2) * (0 + 4) = 4")
    if metodos3 == "Regla Simpson 1/3":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt

            def simpson_13_rule(f, a, b, n):
                h = (b - a) / n
                x = np.linspace(a, b, n + 1)
                y = f(x)
                area = (h / 3) * (np.sum(y[0:-1:2]) + 4 * np.sum(y[1::2]) + y[-1])
                return area

            # Función de ejemplo
            def f(x):
                return np.sin(x)

            # Intervalo de integración
            a = 0
            b = np.pi

            # Número de subintervalos (debe ser par)
            n = 100

            # Asegurarse de que n sea par
            if n % 2 != 0:
                n += 1

            # Calcular el área aproximada utilizando la regla de Simpson 1/3
            area = simpson_13_rule(f, a, b, n)

            # Crear la gráfica de la función y el área aproximada
            x = np.linspace(a, b, 1000)
            y = f(x)

            fig, ax = plt.subplots()
            ax.plot(x, y, label='Función')
            ax.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.3, label='Área Aproximada')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Regla de Simpson 1/3')
            ax.legend()

            # Mostrar el área aproximada en Streamlit
            st.write("Área Aproximada:", area)
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt

            def simpson_13_rule(f, a, b, n):
                h = (b - a) / n
                x = np.linspace(a, b, n + 1)
                y = f(x)
                area = (h / 3) * (np.sum(y[0:-1:2]) + 4 * np.sum(y[1::2]) + y[-1])
                return area

            # Función de ejemplo
            def f(x):
                return np.sin(x)

            # Intervalo de integración
            a = 0
            b = np.pi

            # Número de subintervalos (debe ser par)
            n = 100

            # Asegurarse de que n sea par
            if n % 2 != 0:
                n += 1

            # Calcular el área aproximada utilizando la regla de Simpson 1/3
            area = simpson_13_rule(f, a, b, n)

            # Crear la gráfica de la función y el área aproximada
            x = np.linspace(a, b, 1000)
            y = f(x)

            fig, ax = plt.subplots()
            ax.plot(x, y, label='Función')
            ax.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.3, label='Área Aproximada')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Regla de Simpson 1/3')
            ax.legend()

            # Mostrar el área aproximada en Streamlit
            st.write("Área Aproximada:", area)
            st.pyplot(fig)
            
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            st.write("Supongamos que queremos aproximar el valor de la integral definida de la función f(x) = x^3 en el intervalo [1, 4] utilizando la regla de Simpson 1/3.")
            st.write(" Dividir el intervalo [1, 4] en subintervalos.")
            st.write("Calcular el ancho del subintervalo.")
            st.write("El ancho del subintervalo (h) se calcula dividiendo la longitud total del intervalo entre el número de subintervalos.")
            st.write("En nuestro caso, h = (4 - 1) / 1 = 3.")
            st.write("Evaluar la función en los puntos extremos y en el punto medio del subintervalo.")
            st.write("Evaluamos la función f(x) en los puntos extremos del subintervalo: f(1) = 1 y f(4) = 64.")
            st.write("Además, evaluamos la función en el punto medio del subintervalo: f(2.5) = 15.625.")
            st.write("Aplicar la fórmula de la regla de Simpson 1/3.")
            st.write("La fórmula de la regla de Simpson 1/3 es la siguiente:")
            st.latex(r'Integral aproximada = (h / 3) * (f(a) + 4 * f(medio) + f(b))')
            st.write("Aplicando la fórmula, obtenemos:")
            st.latex(r'Integral aproximada = (3 / 3) * (1 + 4 * 15.625 + 64) = 46.0833')
            st.write("Por lo tanto, utilizando la regla de Simpson 1/3, la aproximación de la integral definida de f(x) = x^3 en el intervalo [1, 4] es igual a 46.0833.")
    if metodos3 == "Regla Simpson 3/8":
        if st.sidebar.checkbox("Código"):
            st.title("***Código***")
            codigo='''
            import numpy as np
            import matplotlib.pyplot as plt

            def simpson_38_rule(f, a, b, n):
                h = (b - a) / n
                x = np.linspace(a, b, n + 1)
                y = f(x)
                area = (3 * h / 8) * (y[0] + 3 * np.sum(y[1:n:3]) + 3 * np.sum(y[2:n:3]) + y[-1])
                return area

            # Función de ejemplo
            def f(x):
                return np.sin(x)

            # Intervalo de integración
            a = 0
            b = np.pi

            # Número de subintervalos (debe ser múltiplo de 3)
            n = 99

            # Asegurarse de que n sea múltiplo de 3
            if n % 3 != 0:
                n = n + 3 - (n % 3)

            # Calcular el área aproximada utilizando la regla de Simpson 3/8
            area = simpson_38_rule(f, a, b, n)

            # Crear la gráfica de la función y el área aproximada
            x = np.linspace(a, b, 1000)
            y = f(x)

            fig, ax = plt.subplots()
            ax.plot(x, y, label='Función')
            ax.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.3, label='Área Aproximada')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Regla de Simpson 3/8')
            ax.legend()

            # Mostrar el área aproximada en Streamlit
            st.write("Área Aproximada:", area)
            st.pyplot(fig)
            '''
            st.code(codigo,language='python')
            #Dowlander
            st.download_button(
                label="Descargar Codigo",
                data=codigo,
                file_name="sistemaEcuaciones.py",
                mime="text/palin")
            #Codigo
            import numpy as np
            import matplotlib.pyplot as plt

            def simpson_38_rule(f, a, b, n):
                h = (b - a) / n
                x = np.linspace(a, b, n + 1)
                y = f(x)
                area = (3 * h / 8) * (y[0] + 3 * np.sum(y[1:n:3]) + 3 * np.sum(y[2:n:3]) + y[-1])
                return area

            # Función de ejemplo
            def f(x):
                return np.sin(x)

            # Intervalo de integración
            a = 0
            b = np.pi

            # Número de subintervalos (debe ser múltiplo de 3)
            n = 99

            # Asegurarse de que n sea múltiplo de 3
            if n % 3 != 0:
                n = n + 3 - (n % 3)

            # Calcular el área aproximada utilizando la regla de Simpson 3/8
            area = simpson_38_rule(f, a, b, n)

            # Crear la gráfica de la función y el área aproximada
            x = np.linspace(a, b, 1000)
            y = f(x)

            fig, ax = plt.subplots()
            ax.plot(x, y, label='Función')
            ax.fill_between(x, y, where=(x >= a) & (x <= b), alpha=0.3, label='Área Aproximada')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Regla de Simpson 3/8')
            ax.legend()

            # Mostrar el área aproximada en Streamlit
            st.write("Área Aproximada:", area)
            st.pyplot(fig)
            
        if st.sidebar.checkbox("Ejemplo"):
            st.title("Ejemplo")
            st.write("Supongamos que queremos aproximar el valor de la integral definida de la función f(x) = x^4 en el intervalo [0, 2] utilizando la regla de Simpson 3/8.")
            st.write("Dividir el intervalo [0, 2] en subintervalos.")
            st.write("Calcular el ancho del subintervalo.")
            st.write("El ancho del subintervalo (h) se calcula dividiendo la longitud total del intervalo entre el número de subintervalos.")
            st.write("En nuestro caso, h = (2 - 0) / 1 = 2.")
            st.write("Evaluar la función en los puntos extremos y en dos puntos adicionales dentro del subintervalo.")
            st.write("Evaluamos la función f(x) en los puntos extremos del subintervalo: f(0) = 0 y f(2) = 16.")
            st.write("Además, evaluamos la función en dos puntos adicionales dentro del subintervalo: f(0.667) = 0.211 y f(1.333) = 3.555.")
            st.write("Aplicar la fórmula de la regla de Simpson 3/8.")
            st.write("La fórmula de la regla de Simpson 3/8 es la siguiente:")
            st.latex(r'Integral aproximada = (3h / 8) * (f(a) + 3 * f(punto1) + 3 * f(punto2) + f(b))')
            st.write("Aplicando la fórmula, obtenemos:")
            st.latex(r'Integral aproximada = (3 * 2 / 8) * (0 + 3 * 0.211 + 3 * 3.555 + 16) = 9.4245')
            st.write("Por lo tanto, utilizando la regla de Simpson 3/8, la aproximación de la integral definida de f(x) = x^4 en el intervalo [0, 2] es igual a 9.4245.")

    #tercera seccion de definiciones
    if metodos3=="Minimos Cuadrados":
        st.title("Minimos Cuadrados")
        texto ="<p style='text-aling:justify;text-justify:inter-word'>El método de minimos cuadados busca la mejor linea o curva de ajuste que minimixe la discrepancia entre los datos observados y estimados para el modelo.</p>"
        texto+="<p style='text-aling:justify;text-justify:inter-word'>Esta discrepancia se mide mediante la suma de los cuadrados de los residuos, que son los residuos,que son las diferencias entre los valores observados y los valores predichos por el modelo.</p>"
        texto+="<p style='text-aling:justify;text-justify:inter-word'>En el caso de una regresion lineal, la formula para encontrar los coeficientes de la linea de ajuste mediante minimos cuadrados.</p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.latex(r'β1 = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)')
        st.latex(r'β0 = ȳ - β1x̄')
        texto = "<p style='text-aling:justify;text-justify:inter-word'>Donde xi y yi son los valores de las variables independiente e dependiente respectivamente, x̄ y ȳ son las medias de los valores de x e y respectivamente.</p>"
        st.markdown(texto,unsafe_allow_html=True)
        texto="<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones:</strong> </p>"
        texto+="<p style='text-aling:justify;text-justify:inter-word'><strong>1-Analisis de regresión:</strong>Se utiliza para ajustar líneas o curvas de regresión a conjuntos de datos y analizar la relación entre variables.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2-Econometria:</strong> Ayuda a estimar los coeficientes en modelos econométricos para analizar relaciones económicas y predecir variables.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>3-Series Temporales:</strong>Permite ajustar modelos para predecir valores futuros en series temporales, como el análisis financiero y el pronóstico de demanda.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>4-Ajuste de curvas:</strong>Se utiliza para ajustar curvas a datos experimentales y analizar el comportamiento de fenómenos físicos o biológicos.</p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodos3=="Descenso Gradiente":
        st.title("Descenso Gradiente")
        texto="<p style='text-aling:justify;text-justify:inter-word'>El descenso del gradiente es un algoritmo iterativo que busca el mínimo de una función mediante la actualización sucesiva de los parámetros en dirección opuesta al gradiente de la función</p>"
        texto+="<p style='text-aling:justify;text-justify:inter-word'> El gradiente representa la dirección de mayor crecimiento de la función, por lo que al moverse en dirección opuesta al gradiente se busca encontrar el mínimo local o global de la función.</p>"
        texto+="<p style='text-aling:justify;text-justify:inter-word'>El algoritmo del descenso del gradiente se basa en la siguiente fórmula de actualización de los parámetros:</p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.latex(r'θi = θi - α * ∂f/∂θi')
        texto = "<p style='text-aling:justify;text-justify:inter-word'>Donde θi representa los parámetros a ser ajustados, α es la tasa de aprendizaje (que determina el tamaño de los pasos en cada iteración) y ∂f/∂θi es el gradiente parcial de la función con respecto al parámetro θi.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones:</strong> </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>1-Aprendizaje Automatico:</strong>Se utiliza en algoritmos de entrenamiento de modelos de aprendizaje automático, como redes neuronales, para ajustar los pesos y minimizar la función de pérdida.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2- Optimizacion de parametroz:</strong>Se aplica para encontrar los valores óptimos de los parámetros en modelos matemáticos y sistemas complejos.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>3-Analisis de datos:</strong>Ayuda a ajustar modelos a conjuntos de datos para realizar predicciones y tomar decisiones basadas en el análisis de datos.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>4-Inteligencia Artificial:</strong>El descenso del gradiente es esencial en algoritmos de entrenamiento de modelos de inteligencia artificial, como algoritmos de aprendizaje profundo (deep learning) y algoritmos de clasificación y reconocimiento de patrones.</p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodos3=="Batch y MiniBatch":
        st.title("Batch")
        texto="<p style='text-aling:justify;text-justify:inter-word'>En el enfoque de batch, todo el conjunto de datos de entrenamiento se procesa en una sola pasada o iteración para actualizar los parámetros del modelo.</p>"
        texto+="<p style='text-aling:justify;text-justify:inter-word'>Esto significa que se calcula el gradiente promedio para todas las muestras de entrenamiento y se actualizan los parámetros en consecuencia.</p>"
        texto+="<p style='text-aling:justify;text-justify:inter-word'>El cálculo del gradiente promedio en el enfoque de batch se realiza mediante la siguiente fórmula:</p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.latex(r'∇θ = (1/N) * Σ ∇L(x, y; θ)')
        texto = "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones:</strong> </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>El enfoque de batch se utiliza en situaciones en las que el tamaño del conjunto de datos de entrenamiento es relativamente pequeño y se dispone de suficiente memoria y recursos computacionales para procesarlo de una sola vez.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Es común en problemas de clasificación y regresión donde se tienen datos suficientes para calcular el gradiente promedio con precisión.</p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.title("Mini Batch")
        texto = "<p style='text-aling:justify;text-justify:inter-word'>En el enfoque de minibatch, el conjunto de datos de entrenamiento se divide en lotes más pequeños llamados minibatches.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>En cada iteración, se procesa un minibatch a la vez y se actualizan los parámetros del modelo en función del gradiente calculado en ese minibatch.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Este enfoque permite el procesamiento en paralelo y el uso eficiente de recursos, especialmente cuando se trabaja con grandes conjuntos de datos.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>La fórmula para el cálculo del gradiente en el enfoque de minibatch es similar a la del enfoque de batch, pero se aplica solo al minibatch actual en lugar de todo el conjunto de datos:</p>"
        st.markdown(texto,unsafe_allow_html=True)
        st.latex(r'∇θ = (1/B) * Σ ∇L(x, y; θ)')
        texto = "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones:</strong> </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>El enfoque de minibatch es ampliamente utilizado en el aprendizaje automático, especialmente cuando se tienen conjuntos de datos grandes o cuando los recursos computacionales son limitados. </p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodos3=="Regla Trapezoidal":
        st.title("Regla Trapezoidal")
        texto = "<p style='text-aling:justify;text-justify:inter-word'>La regla trapezoidal se basa en aproximar el área bajo una curva mediante trapezoides.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>En lugar de calcular exactamente el área, se divide el intervalo de integración en segmentos más pequeños y se estima el área de cada trapezoide formado.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Luego, se suman las áreas de todos los trapezoides para obtener una aproximación del valor de la integral.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>La fórmula general para aplicar la regla trapezoidal a una integral definida es la siguiente:</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>∫[a, b] f(x) dx ≈ ((b - a) / 2n) * [f(a) + 2f(x1) + 2f(x2) + ... + 2f(xn-1) + f(b)]</p>"
        st.markdown(texto,unsafe_allow_html=True)
        texto = "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones:</strong> </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>1-Calculo Numerico:</strong>La regla trapezoidal es una técnica comúnmente utilizada para aproximar integrales definidas cuando no es posible obtener una solución analítica exacta.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2-Analisis de datos:</strong>La regla trapezoidal puede ser utilizada en el análisis y procesamiento de datos para estimar áreas bajo curvas o superficies. </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>3-Simulaciones y modelado:</strong>En simulaciones y modelado matemático, la regla trapezoidal puede ser utilizada para aproximar la integral de funciones que representan fenómenos físicos o procesos complejos.</p>"
        st.markdown(texto,unsafe_allow_html=True)

    elif metodos3=="Regla Simpson 1/3":
        st.title("Regla Simpson 1/3")
        texto = "<p style='text-aling:justify;text-justify:inter-word'>La regla de Simpson 1/3 se basa en aproximar el área bajo una curva mediante segmentos de parábolas.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>En lugar de calcular exactamente el área, se divide el intervalo de integración en segmentos más pequeños y se estima el área de cada parábola formada. </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Luego, se suman las áreas de todas las parábolas para obtener una aproximación del valor de la integral.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>La fórmula general para aplicar la regla de Simpson 1/3 a una integral definida es la siguiente:</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>∫[a, b] f(x) dx ≈ (h / 3) * [f(a) + 4f(x1) + 2f(x2) + 4f(x3) + ... + 2f(xn-2) + 4f(xn-1) + f(b)]</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Donde [a, b] es el intervalo de integración, f(x) es la función que se está integrando, n es el número de segmentos y h es la longitud del segmento.</p>"
        st.markdown(texto,unsafe_allow_html=True)
        texto = "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones</strong> </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>1- Calculo numerico:</strong>Es una técnica comúnmente utilizada para aproximar integrales definidas, especialmente cuando la función integrada tiene una forma suave y bien comportada.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'><strong>2-Analisis de datos:</strong>Se aplica en el análisis y procesamiento de datos para estimar áreas bajo curvas o superficies, especialmente cuando se requiere una mayor precisión en la aproximación que la que proporciona la regla trapezoidal.</p>"
        st.markdown(texto,unsafe_allow_html=True)
    elif metodos3=="Regla Simpson 3/8":
        st.title("Regla Simpson 3/8")
        texto = "<p style='text-aling:justify;text-justify:inter-word'>La regla de Simpson 3/8 es una extensión de la regla de Simpson 1/3 y se utiliza para aproximar el valor de una integral definida utilizando segmentos de curvas cúbicas.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Al igual que la regla de Simpson 1/3, divide el intervalo de integración en segmentos más pequeños y estima el área bajo cada curva cúbica formada.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>La fórmula general para aplicar la regla de Simpson 3/8 a una integral definida es la siguiente:</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>∫[a, b] f(x) dx ≈ (3h / 8) * [f(a) + 3f(x1) + 3f(x2) + 2f(x3) + 3f(x4) + ... + 2f(xn-2) + 3f(xn-1) + f(b)]</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Donde [a, b] es el intervalo de integración, f(x) es la función que se está integrando, n es el número de segmentos y h es la longitud del segmento.</p>"
        st.markdown(texto,unsafe_allow_html=True)
        texto = "<p style='text-aling:justify;text-justify:inter-word'><strong>Aplicaciones:</strong> </p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'>Es importante tener en cuenta que tanto la regla de Simpson 1/3 como la regla de Simpson 3/8 son métodos de aproximación y su precisión aumenta a medida que se utilizan más segmentos.</p>"
        texto += "<p style='text-aling:justify;text-justify:inter-word'> Sin embargo, en casos donde la función integrada tiene propiedades especiales o existe una solución analítica exacta, otras técnicas de integración numérica pueden proporcionar resultados más precisos.</p>"
        st.markdown(texto,unsafe_allow_html=True)



if __name__ == "__main__":
    main()
