from .models import State

def export_fs_as_csv(self):
        """
        Exports FinancialStatement object to CSV
        """
        assert self.state == State.COMPLETED
        all_years = []
        seen = set()
        for item in self.fs.lines:
            for year in item.data.keys():
                if year not in seen:
                    seen.add(year)
                    all_years.append(year)

        with open(self.export_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Label"] + all_years)

            for item in self.fs.lines:
                row = [item.label]
                for year in all_years:
                    value = item.data.get(year, "")
                    row.append(value)
                writer.writerow(row)