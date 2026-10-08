import Foundation

struct DragonForgeStatus: Codable {
    let service: String
    let mode: String
    let training: Bool
    let step: Int?
    let loss: Double?
    let experts: Int?
    let vram_gb: Double?
    let doctor: String
}

@MainActor
final class CompanionModel: ObservableObject {
    @Published var status: DragonForgeStatus?
    @Published var error: String?
    @Published var baseURL = "http://127.0.0.1:8765"

    func refresh() async {
        guard let url = URL(string: baseURL + "/api/v1/status") else {
            error = "Invalid server address"
            return
        }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode) else {
                throw URLError(.badServerResponse)
            }
            status = try JSONDecoder().decode(DragonForgeStatus.self, from: data)
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }
}
