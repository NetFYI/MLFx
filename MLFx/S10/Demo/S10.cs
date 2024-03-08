using Microsoft.ML;

namespace MLFx.S10.Demo
{
    class S10
    {
        public static void Excute()
        {
            MLContext context = new();

            List<InputModel> data = [
                new InputModel {YearsOfExperince = 1, Salary = 39000},
                new InputModel {YearsOfExperince = 1.3F, Salary = 46200},
                new InputModel {YearsOfExperince = 1.5F, Salary = 37700},
                new InputModel {YearsOfExperince = 2, Salary = 43500},
                new InputModel {YearsOfExperince = 2.9F, Salary = 56000},
                new InputModel {YearsOfExperince = 3, Salary = 54000},
                new InputModel {YearsOfExperince = 1, Salary = 2000},
                new InputModel {YearsOfExperince = 4, Salary = 58000}
            ];

            IDataView trainedData = context.Data.LoadFromEnumerable(data);

            var estimatior = context.Transforms.Concatenate("Features", ["YearsOfExperince"]);

            var pipleine = estimatior.Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));

            var model = pipleine.Fit(trainedData);

            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var experince = new InputModel { YearsOfExperince = 10 };
            var result = predictionEngine.Predict(experince);

            Console.WriteLine($"Approx: {result.Salary.ToString()}");

        }
    }
}
