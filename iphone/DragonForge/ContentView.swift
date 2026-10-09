import SwiftUI

struct ContentView: View {
    @StateObject private var model = CompanionModel()

    var body: some View {
        NavigationStack {
            List {
                Section("Connection") {
                    TextField("Server address", text: $model.baseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Button("Refresh") { Task { await model.refresh() } }
                }

                if let status = model.status {
                    Section("Training") {
                        LabeledContent("State", value: status.training ? "Running" : "Idle")
                        LabeledContent("Step", value: status.step.map(String.init) ?? "—")
                        LabeledContent("Loss", value: status.loss.map { String(format: "%.4f", $0) } ?? "—")
                        LabeledContent("Experts", value: status.experts.map(String.init) ?? "—")
                        LabeledContent("VRAM", value: status.vram_gb.map { String(format: "%.2f GB", $0) } ?? "—")
                    }
                    Section("Training Doctor") {
                        LabeledContent("Health", value: status.doctor.capitalized)
                    }
                }

                if let error = model.error {
                    Section("Connection problem") { Text(error) }
                }
            }
            .navigationTitle("DragonForge")
            .task { await model.refresh() }
            .refreshable { await model.refresh() }
        }
    }
}
