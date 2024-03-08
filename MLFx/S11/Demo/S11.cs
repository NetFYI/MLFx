using Microsoft.ML;

namespace MLFx.S11.Demo
{
    class S11
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
                new InputModel {YearsOfExperince = 4, Salary = 58000},
                new InputModel {YearsOfExperince = 7, Salary = 98000},
                new InputModel {YearsOfExperince = 8.5F, Salary = 109000},

            ];

            IDataView trainedData = context.Data.LoadFromEnumerable(data);

            var estimatior = context.Transforms.Concatenate("Features", ["YearsOfExperince"]);

            var pipleine = estimatior.Append(context.Regression.Trainers.Sdca(labelColumnName:"Salary",maximumNumberOfIterations:100));

            var model = pipleine.Fit(trainedData);

            var testDataView = context.Data.LoadFromEnumerable(data);
            var metrics = context.Regression.Evaluate(model.Transform(testDataView), labelColumnName: "Salary");

            Console.WriteLine($"R:2: {metrics.RSquared:0.00}");
            Console.WriteLine($"MA Error: {metrics.MeanAbsoluteError:0.00}");
            Console.WriteLine($"MS Error: {metrics.MeanSquaredError:0.00}");
            Console.WriteLine($"RMS Error: {metrics.RootMeanSquaredError:0.00}");

        }
    }
}
